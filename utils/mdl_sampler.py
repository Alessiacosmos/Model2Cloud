"""
@File           : mdl_sampler.py
@Author         : Gefei Kong
@Time:          : 09.01.2024 19:18
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
sample points from triangle meshes
The main idea refers pyntcloud
"""

from typing import List, Literal

import numpy as np

from utils import mdl_tris


class RandomSampler:
    def __init__(self,
                 meshes_tris: List[dict],
                 sample_mode:Literal["point_density", "point_spacing"]="point_density",
                 pt_param:int or float=10):
        self.meshes_tris = meshes_tris
        self.sample_mode = sample_mode

        if self.sample_mode=="point_density": # pt_param = point_density
            self.n_per_sqm = pt_param
        else: # pt_param = point_spacing
            # calc. point_density based on point_spacing.
            # ref link: https://pro.arcgis.com/en/pro-app/3.1/help/data/las-dataset/work-with-las-dataset-statistics.htm#:~:text=Point%20spacing%20(PS)%20is%20defined,lower%20values%20for%20point%20spacing.
            self.n_per_sqm = int(np.ceil(1./(pt_param**2)))


    def sampling_1mesh(self, mtris:np.ndarray, n_per_sqm:int) -> np.ndarray:
        """
        sample points from a mesh.
        ref code: pyntcloud.samplers.RandomMeshSampler.compute()
        the sampling method refers new algorithm from Prinston (to create more uniform result)
        algorithm: P = (1 - sqrt(r1)) * A + sqrt(r1) * (1 - r2) * B + sqrt(r1) * r2 * C，
        令s,t在区间[0,1]之间取值，随机点Q的值依靠以下算法获得：
            a ← 1−√t;
            b ← (1−s)√t;
            c ← s√t;
            Q ← aA+bB+cC;
        ref link:
            1. https://blog.csdn.net/u014028063/article/details/84314780
            2. http://www.cs.princeton.edu/~funk/tog02.pdf
        :param mtris:       generated mesh_triangels from mesh_vertices and faces (info.). The output of Mesh2Triangle.tris
                            e.g.
                            [(0){"verticies": np.array([[x,y,z],...,[x,y,z]]),
                                 "triangles": np.array([[v1_idx, v2_idx, v3_idx],...,[v1_idx, v2_idx, v3_idx]]),
                                 [optional]"segments",......},
                              ...,
                              (N){(another mesh vertices, triangles, etc.)},
                            ] (N=num_of_meshes_in_the_model)
        :param n_per_sqm:   required number of points per square meter.
        :return:
            res_xyz:        generated (sampling) point clouds. shape=[num_of_point_clouds, 3(xyz)]
                            num_of_point_clouds = ∑np.ceil(mesh[i]_area * n_per_sqm)
        """
        assert len(mtris.shape)==3, \
            f"the input mesh tris shape should be 3-d (shape = [num_tris_in_a_mesh, 3(3 vertices of a tri), 3(xyz)])," \
            f"but {mtris.shape} was gotten."
        assert mtris.shape[-1]>=3, f"each input point coords should include at least xyz info."
        ##########
        # get 3 vertices info.
        ##########
        mtris_v1_xyz = mtris[:, 0, :3]  # shape=[num_tris_in_a_mesh, 3(xyz)]
        mtris_v2_xyz = mtris[:, 1, :3]  # shape=[num_tris_in_a_mesh, 3(xyz)]
        mtris_v3_xyz = mtris[:, 2, :3]  # shape=[num_tris_in_a_mesh, 3(xyz)]

        ##########
        # calc. tris areas to get each tris sampling prob. (num.)
        ##########
        mtris_areas = mdl_tris.tri_area_multi(mtris_v1_xyz, mtris_v2_xyz, mtris_v3_xyz)
        mesh_area = np.sum(mtris_areas)
        probabilities = mtris_areas / mesh_area

        ##########
        # based on mesh area to get the point number need to be sampled
        ##########
        mesh_n = int(np.ceil(mesh_area * n_per_sqm))


        ##########
        # random sampling
        ##########
        random_idx = np.random.choice(
            np.arange(len(mtris_areas)), size=mesh_n, p=probabilities)

        v1_xyz_rand = mtris_v1_xyz[random_idx]
        v2_xyz_rand = mtris_v2_xyz[random_idx]
        v3_xyz_rand = mtris_v3_xyz[random_idx]

        # # (n, 1) the 1 is for broadcasting
        # u = np.random.uniform(low=0., high=1., size=(mesh_n, 1))
        # v = np.random.uniform(low=0., high=1 - u, size=(mesh_n, 1))
        #
        # res_xyz = (v1_xyz_rand * u) + (v2_xyz_rand * v) + ((1 - (u + v)) * v3_xyz_rand) # np.arr, shape=[tri_n, 3]

        t = np.random.uniform(low=0., high=1., size=(mesh_n, 1))
        s = np.random.uniform(low=0., high=1., size=(mesh_n, 1))
        a = 1 - np.sqrt(t)
        b = (1 - s) * np.sqrt(t)
        c = s * np.sqrt(t)
        res_xyz = v1_xyz_rand * a + v2_xyz_rand * b + v3_xyz_rand * c

        return res_xyz


    def sampling_meshes_PD(self) -> np.ndarray:
        """
        sampling points based on a model (including multiple meshes) and the required point_density (PD)
        :return:
            meshes_pts_meshid: sampling result. shape=[Num_of_point_clouds, 3(xyz)]
        """
        meshes_pts_meshid = []
        for mi, mesh_tris in enumerate(self.meshes_tris):
            #############
            # get basic info. -> all vertices of the mesh & the tris this mesh includes
            #############
            mtri_vs = mesh_tris["vertices"] # np.ndarray, shape=[M, 3], 3=[x,y,z]
            # triangles vertices idx, np.ndarray,
            # shape=[N,3], N=number of tris in this face. 3=vertices idx, e.g.[0,1,2]
            mtri_triidxes = mesh_tris["triangles"]
            # get mtris through vertices info. and idx info.
            mtris = mdl_tris.get_tris(mtri_vs, mtri_triidxes)

            #############
            # sample 1 mesh
            #############
            mesh_pts_xyz = self.sampling_1mesh(mtris, self.n_per_sqm)

            #############
            # organize output res.
            # e.g.
            # meshes_pts = list[np.ndarray]
            # for each np.ndarray, shape=[n_sampling_a_mesh, 4]
            # 4=[x,y,z,mi]
            #############
            pts_mesh_id = np.full(shape=(len(mesh_pts_xyz),1), fill_value=mi)
            mesh_pts_xyz_mi =  np.hstack([mesh_pts_xyz, pts_mesh_id])
            meshes_pts_meshid.append(mesh_pts_xyz_mi)

        meshes_pts_meshid = np.vstack(meshes_pts_meshid) # shape=[all_meshes_pt_num, 4(x,y,z,mi)]

        return meshes_pts_meshid

    def add_guassian_noise(self, pts:np.ndarray, sigma:float=0.01) -> np.ndarray:
        pts_xyz = pts[:,:3]
        guass_noises = np.random.normal(0., sigma, size=pts_xyz.shape)
        pts_xyz = pts_xyz + guass_noises
        pts_wnoise = np.hstack([pts_xyz, pts[:,3:]])

        return pts_wnoise


if __name__=="__main__":
    meshes_tris_exp = [{"vertices":np.array([[0,0,1],[0,1,0],[1,1,0],[1,-1,0]])*5,
                        "triangles": np.array([[0,1,2],[0,2,3]])}]

    # sampling
    pt_dense = 20 # unit: points/sqm
    sampler = RandomSampler(meshes_tris_exp, sample_mode="point_density", pt_param=pt_dense)

    pts_withsegid = sampler.sampling_meshes_PD()
    print(f"created pcd: \n{pts_withsegid}")

    # add noises
    pts_withsegid_noise = sampler.add_guassian_noise(pts_withsegid, sigma=0.05)

    # visualization
    # 2d
    import matplotlib.pyplot as plt
    import triangle as tr
    tr.plot(plt.axes(), **meshes_tris_exp[0])
    plt.scatter(pts_withsegid[:,0], pts_withsegid[:,1], s=2, c="blue")
    plt.show()
    plt.close()

    # 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    tris = mdl_tris.get_tris(meshes_tris_exp[0]["vertices"], meshes_tris_exp[0]["triangles"])
    for tri in tris:
        ax.plot3D(tri[:,0], tri[:,1], tri[:,2], "black")
    ax.scatter3D(pts_withsegid[:,0], pts_withsegid[:,1], pts_withsegid[:,2],s=2, c="blue")
    ax.scatter3D(pts_withsegid_noise[:, 0], pts_withsegid_noise[:, 1], pts_withsegid_noise[:, 2], s=2, c="red")
    plt.show()
    plt.close()









