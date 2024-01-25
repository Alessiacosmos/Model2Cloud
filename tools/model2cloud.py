"""
@File           : model2cloud.py
@Author         : Gefei Kong
@Time:          : 12.01.2024 18:08
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
extract point clouds from .obj model
"""

import argparse
import glob
import logging
import os.path
from pprint import pformat
from typing import Literal

import numpy as np
from tqdm import tqdm

from utils.mdl_io import load_obj, create_folder
from utils.mdl_mesh2tri import Mesh2Triangle
from utils.mdl_sampler import RandomSampler

def model2cloud_one(model_path:str,
                    sample_mode:Literal["point_density", "point_spacing"]="point_density",
                    pt_param:int or float=10,
                    noise_sigma:float=-1,
                    isdebug:bool=False):
    # load obj
    vs, es, fs = load_obj(model_path)

    # triangulation
    mesh2tri = Mesh2Triangle(vs, fs)
    mesh_tris = mesh2tri.tris

    # sampling each mesh.
    sampler = RandomSampler(mesh_tris, sample_mode=sample_mode, pt_param=pt_param)
    pts_withsegid = sampler.sampling_meshes_PD()
    if isdebug:
        print(f"created pcd number: {pts_withsegid.shape}")
        print(f"{pts_withsegid[:5,:]}")
        print(f"{np.unique(pts_withsegid[:,-1])} roof planes are included in model {os.path.basename(model_path)}.")

    # add noises
    if noise_sigma>0:
        pts_withsegid_noise = sampler.add_guassian_noise(pts_withsegid, sigma=noise_sigma)
    else:
        pts_withsegid_noise = pts_withsegid

    return pts_withsegid_noise, mesh_tris

def set_logger(cloud_dir_root:str, args_dict:dict):
    ######################
    # log info.
    ######################
    logger = logging.getLogger("model2cloud")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(cloud_dir_root, 'm2c_params.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"PARAMETER ...\n{pformat(args_dict)}")

    return logger

def models2clouds(models_dir:str, clouds_dir:str=None,
                  sample_mode:Literal["point_density", "point_spacing"]="point_density",
                  pt_param:int or float=10,
                  noise_sigma:float=-1):
    """
    get corresponding clouds for multiple models.
    :param models_dir:  directory of saving models | model_path (with model_name.obj)
    :param clouds_dir:  directory of saving generated clouds
    :param sample_mode: "point_density" or "point_spacing"
    :param pt_param:    the corresponding point density or point spacing for determing the point number of sampling.
    :param noise_sigma: sigma value for added gaussian noise.
    :return:
    """
    if models_dir[-4:]==".obj": # the models_dir is the path of a model actually
        models_pathes = [models_dir]
    else: # the models_dir is exactly a directory which covers several models
        models_pathes = glob.glob(os.path.join(models_dir, "*.obj"))

    for mi, model_path in (pbar := tqdm(enumerate(models_pathes), total=len(models_pathes), smoothing=0.9)):
        pbar.set_description(f"{mi}")

        pts, tris = model2cloud_one(model_path,
                                    sample_mode=sample_mode, pt_param=pt_param,
                                    noise_sigma=noise_sigma)

        # save pts
        if clouds_dir is not None:
            savename = os.path.join(clouds_dir, os.path.basename(model_path).replace(".obj", ".txt"))
            np.savetxt(savename, pts, fmt="%.6f,%.6f,%.6f,%d")

def main(args):
    clouds_dir = os.path.join(args.clouds_dir, "clouds/")
    if clouds_dir is not None:
        create_folder(clouds_dir)

    logger = set_logger(cloud_dir_root=args.clouds_dir, args_dict=vars(args))

    models2clouds(args.models_dir,
                  clouds_dir,
                  args.sample_mode,
                  args.pt_param,
                  args.noise_sigma)


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('Model2Cloud')
    parser.add_argument("--models_dir", type=str, default="../test_data/",
                        help="the path or directory for models need to be processed. required.")
    parser.add_argument("--clouds_dir", type=str, default=None,
                        help="the directory for saving generated clouds. "
                             "if no input is received, the clouds won't be saved.")
    parser.add_argument("--sample_mode", type=str, nargs="?",
                        default="point_density", const="point_density", choices=["point_density", "point_spacing"],
                        help="the sampling mode: whether the pt_param is point_density or point_spacing")
    parser.add_argument("--pt_param", type=int or float, default=20, help="the parameter for sampling")
    parser.add_argument("--noise_sigma", type=float, default=-1,
                        help="the sigma value of additional guassian noise (created by normal distribution)"
                             "for generated point clouds")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)

