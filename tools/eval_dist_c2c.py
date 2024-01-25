"""
@File           : eval_dist_c2c.py
@Author         : Gefei Kong
@Time:          : 13.01.2024 19:53
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
evaluation the generated point clouds by cloud-cloud distance using open3d
"""

import argparse
import glob
import os
import logging
from pprint import pformat

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.mdl_eval import cloud_cloud_dist
from utils.mdl_io import load_txt_cloud, load_las_cloud


def get_comp_list(comp_clouds_dir:str, ref_clouds_dir:str, comp_list_path:str) -> (list, list):
    if comp_list_path is not None:
        comp_list = np.loadtxt(comp_list_path, delimiter=" ", dtype="str")
        if len(comp_list.shape)<2: # only 1 cloud will be compared
            comp_list = comp_list.reshape((1, -1))
        comp_clouds_list = [os.path.join(comp_clouds_dir, _[0]) for _ in comp_list]
        ref_clouds_list  = [os.path.join(ref_clouds_dir, _[1]) for _ in comp_list]
    else:
        comp_clouds_list = glob.glob(os.path.join(comp_clouds_dir, "*.txt"))
        ref_clouds_list  = [os.path.join(ref_clouds_dir, os.path.basename(_).replace(".txt", ".las")) for _ in comp_clouds_list]

    return comp_clouds_list, ref_clouds_list


def set_logger_eval(log_dir:str, args_dict:dict)->logging.Logger:
    ######################
    # log info.
    ######################
    logger = logging.getLogger("model2cloud")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'm2c_params_eval.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"EVAL PARAMETER ...\n{pformat(args_dict)}")

    return logger

def eval_generated_clouds(comp_clouds_dir:str, ref_clouds_dir:str, comp_list_path:str):
    # get the list of clouds need to be considered
    comp_clouds_list, ref_clouds_list = get_comp_list(comp_clouds_dir, ref_clouds_dir, comp_list_path)

    evalres_all = []
    for ci, (comp_cloud_path, ref_cloud_path) in (pbar := tqdm(enumerate(zip(comp_clouds_list, ref_clouds_list)),
                                                               total=len(comp_clouds_list), smoothing=0.9)):
        # get the ID. of the current point cloud
        cloud_id = os.path.basename(ref_cloud_path)[:-4]

        #####
        # load clouds
        #####
        # load comp data
        comp_cloud = load_txt_cloud(comp_cloud_path)
        # load ref data
        ref_cloud = load_las_cloud(ref_cloud_path)


        #####
        # eval
        #####
        dist_mean, dist_std, dist_min, dist_max = cloud_cloud_dist(comp_cloud, ref_cloud)
        evalres_all.append([cloud_id, dist_mean, dist_std, dist_min, dist_max])


    evalres_all = pd.DataFrame(evalres_all, columns=["id", "d_mean", "d_std", "d_min", "d_max"])

    # calculate the mean evalres for the dataset.
    evalres_all_mean = evalres_all.iloc[:,1:].mean()
    evalres_all_mean["id"] = "mean"
    evalres_all.loc["mean"] = evalres_all_mean

    return evalres_all

def main_eval(args):
    args.comp_dir = os.path.dirname(args.comp_dir) if args.comp_dir[-1]=="/" else args.comp_dir
    args.log_dir = os.path.dirname(args.comp_dir) if args.log_dir is None else args.log_dir

    logger = set_logger_eval(log_dir=args.log_dir, args_dict=vars(args))
    evalres_all = eval_generated_clouds(args.comp_dir, args.ref_dir, args.comp_list_path)

    # save mean result to logger:
    logger.info(f"eval_mean_res: \n{evalres_all.loc['mean']}")
    if args.save_res:
        save_name = os.path.join(args.log_dir, "eval_res.csv")
        evalres_all.to_csv(save_name, index=False)


def parse_args_eval():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('Model2Cloud')
    parser.add_argument("--comp_dir", type=str, default="../test_data/test_out/clouds/",
                        help="the directory for generated clouds. required.")
    parser.add_argument("--ref_dir", type=str, default="../test_data/",
                        help="the directory for reference clouds. required.")
    parser.add_argument("--comp_list_path", type=str, default=None,
                        help="the path of list including the point clouds names for comparison and ref. "
                             "list example: "
                             "  ['1_com.txt 1_ref.las', "
                             "   '2_com.txt 2_ref.las', "
                             "   ...] "
                             "each path in the list will be organized as 'comp_dir'+'list[i]'. "
                             "default=None. If None is recieved, all '.txt' file in comp_dir will be considered.")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="the directory for logger. if none, use the upper directory of comp_dir as the logger dir.")
    parser.add_argument("--save_res", type=bool, default=False,
                        help="whether save the eval result to file. "
                             "If True, the eval result will be saved as .csv file in the log_dir.")

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args_eval()
    main_eval(args)
