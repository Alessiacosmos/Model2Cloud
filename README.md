# Model2Cloud
Generate point clouds based on input 3D building model.  
The code repository for manuscript *"Automatic Generation of 3D roof TRaining Dataset for Building Roof Segmentation from ALS Point Clouds"*

## Usage
### 1. Generate clouds
#### 1.1 Generate from single .obj file
use `tools/model2cloud.py` to achieve the generation of point clouds.
```shell
$ python3 -m tools.model2cloud --models_dir test_data/ --clouds_dir test_data/test_out/ --sample_mode point_density --pt_param 20 --noise_sigma 0.02
```
<span id="params_1-1">parameters</span>
```shell
$ python3 -m tools.model2cloud --help
usage: Model2Cloud [-h] --models_dir MODELS_DIR [--clouds_dir CLOUDS_DIR] [--sample_mode [{point_density,point_spacing}]]
                   [--pt_param PT_PARAM] [--noise_sigma NOISE_SIGMA]

optional arguments:
  -h, --help            show this help message and exit
  --models_dir MODELS_DIR
                        the path or directory for models need to be processed.
  --clouds_dir CLOUDS_DIR
                        the directory for saving generated clouds. if no input is received, the clouds wont be saved.
  --sample_mode [{point_density,point_spacing}]
                        the sampling mode: whether the pt_param is point_density or point_spacing
  --pt_param PT_PARAM   the parameter for sampling
  --noise_sigma NOISE_SIGMA
                        the sigma value of additional guassian noise (created by normal distribution)for generated point
                        clouds
```

#### 1.2 Generate from .citygml file
use `tools/model2cloud_citygml.py` to achieve the generation of point clouds for a series of 3D buildings in a region.
```shell
$ python3 -m tools.model2cloud_citygml --models_dir test_data/gml_files/Altenbach --clouds_dir test_data/gml_objs/Altenbach --t_rm_abnormal_face 0.3 --sample_mode point_density --pt_param 20 --noise_sigma 0.02
```
parameters are the same as [1.1 parameters](#params_1-1)


### 2. cloud-cloud distance to evaluate the quality of generated clouds
use `tools/eval_dist_c2c.py` to achieve the cloud-cloud distance evaluation.
```shell
$ python3 -m tools.eval_dist_c2c --comp_dir test_data/test_out/clouds/ --ref_dir test_data/ --comp_list_path test_data/test_comp_list.txt --save_res True
```
paramters
```shell
$ python3 -m tools.eval_dist_c2c --help
usage: Model2Cloud [-h] [--comp_dir COMP_DIR] [--ref_dir REF_DIR] [--comp_list_path COMP_LIST_PATH] [--log_dir LOG_DIR]
                   [--save_res SAVE_RES]

optional arguments:
  -h, --help            show this help message and exit
  --comp_dir COMP_DIR   the directory for generated clouds. required.
  --ref_dir REF_DIR     the directory for reference clouds. required.
  --comp_list_path COMP_LIST_PATH
                        the path of list including the point clouds names for comparison and ref. list example:
                        ['1_com.txt 1_ref.las', '2_com.txt 2_ref.las', ...] each path in the list will be organized as
                        'comp_dir'+'list[i]'. default=None. If None is recieved, all '.txt' file in comp_dir will be
                        considered.
  --log_dir LOG_DIR     the directory for logger. if none, use the upper directory of comp_dir as the logger dir.
  --save_res SAVE_RES   whether save the eval result to file. If True, the eval result will be saved as .csv file in the
                        log_dir.
```

## Requirements
The following requirements are necessary for Model2Cloud:
- laspy
- triangle

Other potential requirements can be found in [requriements_m2c.txt](requirements_m2c.txt) and according to the compilation errors.
