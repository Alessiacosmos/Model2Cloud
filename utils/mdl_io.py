"""
@File           : mdl_io.py
@Author         : Gefei Kong
@Time:          : 09.01.2024 21:28
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
i/o related functions
"""

import os

import laspy
import numpy as np
import open3d as o3d
import pandas as pd

from utils import mdl_CityGML2OBJs as citygml2obj
from utils.mdl_utils import calc_face_normal



def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("create folder: ", folder_path)
    else:
        print(f"folder {folder_path} already exists.")


def load_las_cloud(filepath:str)->np.ndarray:
    """
    load point cloud data from .las file
    :param filepath:
    :return:
            point clouds, shape=[n, 3]
            # m: the number of planes
            n: the number of points in a plane
            3: xyz
    """
    cloud_ij = laspy.read(filepath)
    xyz_ij = np.vstack((cloud_ij.x, cloud_ij.y, cloud_ij.z)).transpose()

    return xyz_ij

def load_txt_cloud(filepath:str, delimiter:str=",")->np.ndarray:
    """
    load point cloud data from .txt file
    :param filepathes:
    :return:
            point clouds, shape=[n, 4]
            # m: the number of planes
            n: the number of points in a plane
            4: xyz[plane_id]
    """
    cloud = np.loadtxt(filepath, delimiter=delimiter)
    return cloud


def xyz2o3d_cloud(xyz:np.ndarray):
    assert xyz.shape[1]==3, f"only accept 2-d xyz arr whose shape is [N,3], but {xyz.shape} was gotten."

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


def load_obj(obj_path:str) -> (np.ndarray, np.ndarray, list):
    with open(obj_path, 'r') as f:
        lines = f.readlines()

    vs, edges, faces = [], set(), []
    for f in lines:
        vals = f.strip().split(' ')
        # print('vals: ', vals)
        if vals[0] in ['#', 'vn']:
            continue
        elif vals[0] == 'v':
            # v in obj files: x, y, z
            vs_i = [vals[1], vals[2], vals[3]]
            vs.append(vs_i)
        else:
            face_data = np.array(vals[1:])
            face_data = np.array(list(np.char.split(face_data, sep='//')),
                                 dtype='int') - 1  # [[vertices_No., face_normal_No.]] # shape: (n, 2)
            # print("face data: ", face_data)
            face_i = face_data[:, 0].reshape(-1)  # shape: (n, 1)
            faces.append(face_i.tolist())

            edge_v = face_data[:, 0].reshape(-1, 1)  # shape: (n, 1)
            # face_No = face_data[:,1].reshape(-1, 1) # shape: (n, 1)
            idx = np.arange(len(edge_v)) - 1
            cur_edge = np.concatenate([edge_v, edge_v[idx]], -1)  # shape: (n,2) [[start_v, end_v], ...]
            [edges.add(tuple(sorted(e))) for e in cur_edge]

    vs = np.array(vs, dtype=np.float64)
    # vs[:, 1] = vs[:, 1] * (-1)  # -y -> y
    edges = np.array(list(edges))

    return vs, edges, faces


def save_obj_str(out_filename:str, vs_str:np.ndarray, fs_str:list):
    """
        save vertices and faces (have been organized as str) to obj file
        when faces ARE organized and start from 1
        :param out_filename:
        :param vs:
        :param fs:
        :return:
        """
    # save vs and vn info. at first
    np.savetxt(out_filename, vs_str, delimiter=' ', fmt='%s')
    # then save face info. after vs and vn info.
    with open(out_filename, 'a') as out_file:
        out_file.writelines(fs_str)


def save_obj(out_filename:str, vs:np.ndarray, fs:list):
    """
    save vertices and faces to obj file
    when faces are NOT organized, such as faces index starts from 0
    :param out_filename:
    :param vs:  vertices
    :param fs:  faces
    :return:
    """
    # organize vertices to 'v <x> <y> <z>'
    v_head = np.full(shape=(vs.shape[0], 1), fill_value='v', dtype='str')
    vs_str = np.char.mod("%.6f", vs)
    vs_str = np.hstack([v_head, vs_str])

    # organize faces
    fs_str = []
    is_start_1 = min([_ for f in fs for _ in f])
    # print("is_start_1: ", is_start_1)
    for i in range(len(fs)):
        if is_start_1:
            f_str = "f " + " ".join(map(str, fs[i])) + "\n"
        else:
            f_str = "f " + " ".join(map(str, np.array(fs[i]) + 1)) + "\n"
        fs_str.append(f_str)

    save_obj_str(out_filename, vs_str, fs_str)


def load_obj_multiblds(obj_path:str, save_single_obj:bool=True, **kwargs) -> dict:
    # ignore the warning when force creating an np.arr from a list with unequal length sublists like this -> [[1,2],[1,2,3]]
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


    with open(obj_path, 'r') as f:
        lines = f.readlines()

    # save multiple buildings vertices, edges and faces.
    vs_all, faces_sep = [], {}
    for f in lines:
        if len(f)==0: # blank line
            continue

        vals = f.strip().split(' ')
        # print('vals: ', vals)
        if vals[0] in ['#', 'vn', '']:
            continue
        elif vals[0] == 'v':
            # v in obj files: x, y, z
            vs_i = [vals[1], vals[2], vals[3]]
            vs_all.append(vs_i)
        elif vals[0] == 'o':
            bld_id = vals[1]
            faces_sep[bld_id] = []
        else: # start to sort face info
            face_data = np.array(vals[1:])
            face_data = np.array(list(np.char.split(face_data, sep='//')),
                                 dtype='int') - 1   # shape: (n, 1)
            # print("face data: ", face_data)
            face_i = face_data[:, 0].reshape(-1)  # shape: (n, )
            faces_sep[bld_id].append(face_i.tolist())

    # seperate each building and get roof model
    vs_pd = pd.DataFrame(vs_all, dtype=np.float64, columns=['x','y','z'])

    rf_sep = {} # all roof info.
    for bi, (bld_k, bld_fs) in enumerate(faces_sep.items()):
        # if bi > 10: break
        # if bld_k != "DEBW_DEBWL0010000Cipx":
        #     continue
        rf_sep[bld_k] = {}

        # judge whether the single obj has created or not; and whether refresh the existed single obj file.
        if save_single_obj:
            save_sobj_dir = kwargs.get("save_dir", None)
            refresh       = kwargs.get("refresh_savedobj", False)
            if save_sobj_dir is None:
                raise NameError("argument 'save_dir' for saving generated single objs are expected but not defined, "
                                "when save_single_obj is True.")
            save_obj_path = os.path.join(save_sobj_dir, f"{bld_k}.obj")

            if (not refresh) and (os.path.exists(save_obj_path)):
                vs, es, fs = load_obj(save_obj_path)
                rf_sep[bld_k]["vertices"] = vs
                rf_sep[bld_k]["faces"] = fs

                continue

        #########
        # get the vertices information for this building
        #########
        bld_vs = vs_pd.iloc[np.unique([_ for fi in bld_fs for _ in fi]), :]

        #########
        # get roof surface
        #########
        bld_f_nz_h = [] # bld_f's normal-z value and mean height
        for bld_f in bld_fs: # for each face, calculate its normal
            # 1. calculate each face's normal-z and mena height
            fn = np.round(calc_face_normal(vs_pd.iloc[bld_f, :]), 2)  # round to 0.01 to let some very small value like 1e-10 = 0
            bld_f_nz_h.append([fn[2], np.mean(vs_pd.iloc[bld_f, 2])]) # [normal-z value, mean height]

        # 2. remove wall's and ground faces.
        #    wall: normal-z = 0. ground: height (z-value) is low.
        #    remove wall's faces, roof surfaces are the left surfaces whose mean heights are higher.
        bld_f_nz_h = np.asarray(bld_f_nz_h)
        bld_wallf_hp25 = np.percentile(bld_f_nz_h[np.abs(bld_f_nz_h[:,0])==0, 1], 25) # 25% percentile as the wall's mean height to seperate roof and ground

        # 3. get roof surfaces:
        # condition 1: np.abs(bld_f_nz_h[:,0])!=0      ===>  not wall's normal-z
        # condition 1: bld_f_nz_h[:,1]>bld_wallf_hp25  ===>  in the left surfaces, select them with higher z-value.
        bld_rfs = np.asarray(bld_fs)[(np.abs(bld_f_nz_h[:,0])!=0) & (bld_f_nz_h[:,1]>bld_wallf_hp25)].tolist()
        # 4. get roof surface vertices:
        bld_rvs = bld_vs.loc[np.unique([_ for rf in bld_rfs for _ in rf])]

        #########
        # 5. re-ogranize faces idx to start from 0
        #########
        # re-organize bld_rvs with new idx from 0 to N_vs
        # the new dataframe includes 4 columns: idx_new ([1, N_vs]), x, y, z
        # and the index of this dataframe is still the unchanged faces idx not from 0.
        bld_rvs_ridx = bld_rvs.rename_axis('idx_o').reset_index()
        bld_rvs_ridx = bld_rvs_ridx.rename_axis('idx_new').reset_index().set_index('idx_o')
        bld_rvs_ridx["idx_new"] += 1 # [0,N_vs)->[1,N_vs]
        bld_rfs = [bld_rvs_ridx.loc[rf, "idx_new"].values.tolist() for rf in bld_rfs]
        # save them to a dict for the final output
        rf_sep[bld_k]["vertices"] = bld_rvs.values # np.ndarray, shape=[n,3]
        rf_sep[bld_k]["faces"] = bld_rfs           # list

        if save_single_obj:
            save_obj(save_obj_path, rf_sep[bld_k]["vertices"], rf_sep[bld_k]["faces"])
            # print("save refresh")

        # print("bld_vs: \n", bld_vs)
        # print("bld_f_nz_h:\n", bld_f_nz_h)
        # print("fin vs: \n", rf_sep[bld_k]["vertices"])

    return rf_sep

def cvt_citygml2obj(gml_path:str, save_dir:str)->str:
    """
    use CITYGML2OBJs_v2 to achieve the conversion from citygml file to obj file.
    repo. link: https://github.com/tum-gis/CityGML2OBJv2/
    And we modified it as a module.
    :param gml_path: gml path with file_basename
    :param save_dir: the dir saving converted .obj file.
    :return:
    """
    # set necessary parameters for citygml2obj module
    citygml2obj.ARGS['directory'] = gml_path
    citygml2obj.ARGS['results'] = save_dir

    obj_dir, gml_filename = citygml2obj.main()

    save_path = os.path.join(obj_dir, f"{gml_filename}.obj")

    return save_path



