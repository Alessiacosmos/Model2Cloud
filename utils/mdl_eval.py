"""
@File           : mdl_eval.py
@Author         : Gefei Kong
@Time:          : 13.01.2024 19:53
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
evaluation
"""
import numpy as np
import open3d

from utils.mdl_io import xyz2o3d_cloud


def cloud_cloud_dist(comp_cloud:np.ndarray, ref_cloud:np.ndarray)->(float, float, float):
    """
    calculate cloud_cloud_dist by computes for each point in the source point cloud the distance to the closest point in the target point cloud.
    ref introduction link: https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html
    :param comp_cloud:
    :param ref_cloud:
    :return:
    """
    comp_xyz = xyz2o3d_cloud(comp_cloud[:, :3])
    ref_xyz  = xyz2o3d_cloud(ref_cloud[:, :3])

    dists = comp_xyz.compute_point_cloud_distance(ref_xyz)

    dist_mean, dist_std, dist_min, dist_max = np.mean(dists), np.std(dists), np.min(dists), np.max(dists)

    return dist_mean, dist_std, dist_min, dist_max


if __name__=="__main__":
    # clouds pathes
    # compare
    comp_cloud_path = "../test_data/test_out/clouds/10444144_rfstruct.txt"
    # ref
    ref_cloud_path  = "../test_data/10444144.las"

    # load clouds
    from utils.mdl_io import load_txt_cloud, load_las_cloud
    # load comp data
    comp_cloud = load_txt_cloud(comp_cloud_path)
    # load ref data
    ref_cloud  = load_las_cloud(ref_cloud_path)

    dist_mean, dist_std, dist_min, dist_max = cloud_cloud_dist(comp_cloud, ref_cloud)
    print(f"dist_mean={dist_mean}, dist_std={dist_std}, dist_min={dist_min}, dist_max={dist_max}")
