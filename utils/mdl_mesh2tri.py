"""
@File           : mdl_mesh2tri.py
@Author         : Gefei Kong
@Time:          : 09.01.2024 20:08
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
a surface (mesh) to triangles
"""
import warnings

import numpy as np
import triangle as tr

from utils.mdl_io import load_obj
from utils.mdl_utils import get_polygonMBR_min_edge


class Mesh2Triangle:
    def __init__(self, meshes_v:np.ndarray or list,
                 meshes_f:list=None, rm_smallface:float=-1):
        """
        :param meshes_v: vertices matrix. shape=[M,3]
                         M: number of vertex.
                         3: [x,y,z]
        :param meshes_f: face list. e.g., [0,1,2,3,4]
        :param rm_smallface: whether remove the face who is very small and can be regarded as abnormal faces. -1 = won't remove; >0: remove based on the min_length_MBR; unit: m
        :return:
            self.tris:   a list of tris. len(self.tris) = len(meshes_f), the number of mesh faces
                         Each tri is a dictionary and includes vertices and triangles info.
                         e.g. {"verticies": np.array([[x,y,z],...,[x,y,z]]),
                               "triangles": np.array([[v1_idx, v2_idx, v3_idx],...,[v1_idx, v2_idx, v3_idx]]),
                               [optional]"segments",......}
        """
        meshes_v_sep = self.sep_v_meshes(meshes_v, meshes_f)
        self.meshes_v = self.rm_abnormal_faces(meshes_v_sep, rm_smallface)
        self.meshes_e = self.get_edges(self.meshes_v)
        # self.meshes_f = meshes_f

        self.add_v_tag = False # tag whether additional vertices are added (if True, implict the wrong annotation and this roof is better to be passed.)

        self.tris = self.crt_constrained_delaunay_meshes()

    def sep_v_meshes(self, meshes_v, meshes_f) -> list:
        # seperate the vertex array based on the face info.
        # if meshes_v input as 2-d array, including all vertex information of the whole structure and don't seperated based on faces.
        if len(meshes_v[0].shape) == 1:
            assert meshes_f is not None, \
                "in the situation mesh_vertex info. doesn't split based on faces, the face info. should be input."
            meshes_v = [meshes_v[f, :] for f in meshes_f]

        return meshes_v

    def rm_abnormal_faces(self, meshes_v_sep, rm_smallface):
        """
        remove abnormal faces in the meshes.
        abnormal faces:
            1. which are not face actually (include <=2 vertices)
            2. which have a very very small min_edge length for their MBRs, meaning that this face is very small and is errorly annotated.
                e.g. the first faces in Altenbach/singles/DEBW_DEBWL0010000CitJ & DEBW_DEBWL0010000CisO: and the generated points' number of these faces are very very limited, lower than 10.
        :param meshes_v_sep:      each face's vertices (has been separated by face information)
        :param rm_smallface:      the threshold to remove the 2nd-type abnormal faces (the min_edge_len of a normal face's MBR)
        :return:
               meshes_v_clean:    cleaned meshes
        """
        meshes_v_clean = []

        if len(meshes_v_sep)<=1: # only 1 face, which won't have problem.
            return meshes_v_sep

        for mesh_v in meshes_v_sep:
            mesh_min_edge_len = get_polygonMBR_min_edge(mesh_v)
            if mesh_min_edge_len >= rm_smallface:
                meshes_v_clean.append(mesh_v)

        return meshes_v_clean

    def get_edges(self, meshes_v) -> list:
        meshes_edges = []
        for mesh_v in meshes_v:
            mesh_f = np.arange(len(mesh_v))
            mesh_edges = np.vstack([mesh_f, np.roll(mesh_f, -1)]).T
            meshes_edges.append(mesh_edges)
        return meshes_edges

    def crt_constrained_delaunay_meshes(self) -> list:
        """
        create constrained Delaunay triangulation for all surfaces
        :return:
        """
        # e.g. input of meshes_v
        # [(mesh 1)[[x,y,z],
        #           ...,
        #           [x,y,z]],
        #  ...,
        #  (mesh N)[[x,y,z],
        #           ...,
        #           [x,y,z]]]

        assert len(self.meshes_v)==len(self.meshes_e), \
            f"the input number of meshes vertex info. and edge info. should be the same, " \
            f"but {len(self.meshes_v)} mesh (for vertex) and {len(self.meshes_e)} meshes (for edges) are obtained."


        tris = []
        for mi, mesh_v in enumerate(self.meshes_v):
            mesh_es = self.meshes_e[mi]

            tri_i = self.crt_constrained_delaunay_1mesh(mesh_vertices=mesh_v,
                                                        mesh_edges=mesh_es)

            if len(tri_i["vertices"]) != len(mesh_v):
                # the number of vertices might not equal to the input vertices' number, because of the wrong annotation. In this case, raise an warning.
                warnings.warn(f"Additional vertices are added when triangulation: "
                              f"original: {len(mesh_v)}, new: {len(tri_i['vertices'])}. "
                              f"The input polygon exists errors.")
                self.add_v_tag = True

            tri_i["vertices"] = mesh_v
            tris.append(tri_i)

        return tris

    def crt_constrained_delaunay_1mesh(self,
                                       mesh_vertices: np.ndarray or list,
                                       mesh_edges: np.ndarray or list) -> dict:
        """
        create constrained delaunay triangulation.
        the constraints are the boundary segments of the polygon (mesh)
        :param mesh_vertices: vertices matrix. shape=[M,3] or [M,2]
                              M: number of vertex.
                              3: [x,y,z]; 2: [x,y]
        :param mesh_edges:    edge matrix. shape=[x,2]
                              x: number of edges, usually = M
                              2: [start_v_idx, end_v_idx]
        :param mesh_face:     face list. e.g., [0,1,2,3,4]
        :return:
               tri:           created Delaunay triangles supported by "triangle" packages.
                              dict includes:
                                'vertices':         (only includes [x,y] info. so should be considered with the meshes_v)
                                'vertex_markers':
                                'triangles':        triangles
                                'segments':
                                'segment_markers':
        """
        if isinstance(mesh_vertices, list):
            mesh_vertices = np.asarray(mesh_vertices)

        if isinstance(mesh_edges, list):
            mesh_edges = np.asarray(mesh_edges)

        # create Delaunay triangulation
        mesh_data = dict(vertices=mesh_vertices[:,:2], segments=mesh_edges)
        tri = tr.triangulate(mesh_data, "p")

        return tri


if __name__=="__main__":
    # DEBW_DEBWL0010000Cirl
    # test rm_abnormal_face: DEBW_DEBWL0010000CitJ; DEBW_DEBWL0010000CisO DEBW_DEBWL0010000Civg DEBW_DEBWL0010000Citd (whose abnoraml face can be saved.)
    model_path = "/home/gefeik/PhD-work/roofstructreDataset/codes_rf_Dataset/test_data/gml_objs/Altenbach/singles/DEBW_DEBWL0010000Civg.obj" # r"../test_data/10444144_rfstruct.obj"

    vs, es, fs = load_obj(model_path)
    vs = vs - np.mean(vs, axis=0)
    print(vs, fs)

    #######
    # visualize faces. to check whether the input is right
    #######
    # import matplotlib.pyplot as plt
    # colors = ["black", "green", "yellow", "red"]
    # plt.scatter(vs[:,0], vs[:,1], s=10, c="b")
    # for fi, f in enumerate(fs):
    #     plt.plot(vs[f,0], vs[f,1], c=colors[fi%len(colors)])
    # plt.show()
    # plt.close()

    mesh2tri = Mesh2Triangle(vs, fs, rm_smallface=0.5)
    tris = mesh2tri.tris
    print(f"vs:\n{vs-np.min(vs, axis=0)}")
    print(f"es:\n{es}")
    print(f"fs:\n{fs}")

    # check the created tris with visual.
    import matplotlib.pyplot as plt
    print(f"tris: ({len(tris)})\n")
    for i, tri in enumerate(tris):
        print(f"tri_{i}(3d)\n{tri}")
        tri_2d = tri.copy()
        tri_2d["vertices"] = tri["vertices"][:,:2]
        print(f"tri_{i}(2d)\n{tri_2d}")
        tr.plot(plt.axes(), **tri_2d)
        plt.title(f"tri_{i}")
        plt.show()
        plt.close()




