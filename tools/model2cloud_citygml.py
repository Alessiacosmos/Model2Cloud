"""
@File           : model2cloud_citygml.py
@Author         : Gefei Kong
@Time:          : 25.01.2024 22:47
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""
import argparse
import logging
import os
from pprint import pformat
from typing import Literal

from tools.model2cloud import models2clouds
from utils.mdl_io import create_folder, cvt_citygml2obj, load_obj_multiblds


def set_logger(cloud_dir_root:str, args_dict:dict):
    ######################
    # log info.
    ######################
    logger = logging.getLogger("model2cloud_citygml")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(cloud_dir_root, f'm2c_params_{os.path.basename(cloud_dir_root)}.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"PARAMETER ...\n{pformat(args_dict)}")

    return logger


def models2clouds_citygml(models_dir:str, clouds_dir:str=None,
                          t_rm_abnormal_face:float=-1,
                          sample_mode:Literal["point_density", "point_spacing"]="point_density",
                          pt_param:int or float=10,
                          noise_sigma:float=-1):

    ###########
    # citygml 2 obj file
    ###########
    # 1. convert .gml file to .obj file.
    obj_dir = os.path.dirname(clouds_dir)
    save_obj_multi_path = cvt_citygml2obj(models_dir, obj_dir) # the save path of .obj file including multiple buildings
    # 2. convert .obj file including multiple buildings to a series of single .obj files.
    obj_single_savedir = os.path.join(obj_dir, "singles")
    create_folder(obj_single_savedir)

    rf_sep = load_obj_multiblds(save_obj_multi_path, save_single_obj=True, save_dir=obj_single_savedir,
                                refresh_savedobj=False)

    # for rf_key, rf_value in rf_sep.items():
    #     rf_vs = rf_value["vertices"]
    #     rf_fs = rf_value["faces"]


    models2clouds(obj_single_savedir,
                  clouds_dir,
                  t_rm_abnormal_face,
                  sample_mode,
                  pt_param,
                  noise_sigma)


def main(args):
    clouds_dir = os.path.join(args.clouds_dir, "clouds")

    if args.noise_sigma > 0:
        clouds_dir = f"{clouds_dir}_noise={args.noise_sigma}"
    if args.t_rm_abnormal_face > 0:
        clouds_dir = f"{clouds_dir}_tabf={args.t_rm_abnormal_face}"

    if args.clouds_dir is not None:
        create_folder(clouds_dir)

    logger = set_logger(cloud_dir_root=args.clouds_dir, args_dict=vars(args))

    models2clouds_citygml(args.models_dir,
                          clouds_dir,
                          args.t_rm_abnormal_face,
                          args.sample_mode,
                          args.pt_param,
                          args.noise_sigma)

def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('Model2Cloud_CityGML')
    parser.add_argument("--models_dir", type=str, default="../test_data/gml_files/Altenbach",
                        help="the path or directory for models need to be processed. required.")
    parser.add_argument("--clouds_dir", type=str, default="../test_data/gml_objs/Altenbach", # None,
                        help="the directory for saving generated clouds. "
                             "if no input is received, the clouds won't be saved.")
    parser.add_argument("--t_rm_abnormal_face", type=float, default=-1,
                        help="the threshold to remove abnormal face before triangulation:"
                             "(1) whose vertices number <=2 or (2) whose MBR's min edge length <= this threshold."
                             "if default (-1), no face will be removed.")
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