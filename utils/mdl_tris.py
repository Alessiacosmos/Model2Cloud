"""
@File           : mdl_tris.py
@Author         : Gefei Kong
@Time:          : 10.01.2024 17:22
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
triangles related functions
"""
import numpy as np


def get_a_tri(tri_idx:np.ndarray, all_vertices:np.ndarray):
    tri = all_vertices[tri_idx, :]
    return tri

def get_tris(all_vertices:np.ndarray, tris_idxes:np.ndarray)->np.ndarray:
    tris = np.apply_along_axis(get_a_tri, axis=1, arr=tris_idxes, all_vertices=all_vertices)
    return tris # shape = [num_tris_in_a_mesh, 3(3 vertices of a tri), 3(xyz)]

def tri_area_multi(v1, v2, v3):
    """ v1, v2, v3 are (N,3) arrays. Each one represents the vertices
    such as v1[i], v2[i], v3[i] represent the ith triangle
    code from: pyntcloud.geometry.areas.triangle_area_multi()
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1,
                                         v3 - v1), axis=1)

if __name__=="__main__":
    vs_exp   = np.array([[0,0,0],[0,1,0],[1,1,0],[1,-1,0]])
    tris_exp = np.array([[0,1,2],[0,2,3]])

    tri = get_a_tri(tris_exp[0,:], vs_exp)
    print(tri)

    tris = get_tris(vs_exp, tris_exp)
    print(tris)
    print(tris[:,0,:])

    ########
    # test whether can get right area result of multi tris
    ########
    tris_v1 = tris[:, 0, :] # shape=[num_tris_in_a_mesh, 3(xyz)/2(xy)]
    tris_v2 = tris[:, 1, :]  # shape=[num_tris_in_a_mesh, 3(xyz)/2(xy)]
    tris_v3 = tris[:, 2, :]  # shape=[num_tris_in_a_mesh, 3(xyz)/2(xy)]
    tris_area = tri_area_multi(tris_v1, tris_v2, tris_v3)
    print(f"tris_area: {tris_area}")