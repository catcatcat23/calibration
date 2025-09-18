import glob
import pickle
import pandas as pd
import torch
import time
import os
import numpy as np
import random
import sys

from utils.utilities import TransformImage, split
from utils.metrics import multiclass_multimetric, accuracy, evaluate_val_resultsnew
from scipy.special import softmax
from copy import deepcopy
from tqdm import tqdm
from tools import GradCAM
from algo_test_utils import get_test_all_csvs, get_test_df
from train_data import get_train_csv
from torchvision import transforms
from dataloader.image_resampler_pins import LUT_VAL

import pickle
import gc
import matplotlib.pyplot as plt
import warnings
import re
warnings.filterwarnings('ignore')
import argparse
import cv2

parser = argparse.ArgumentParser(description='Run inference to check res')   # padgroupv1.22.1tp2select  singlepinpadv2.15tp1select 
parser.add_argument('--version_folder', default='singlepinpadv2.17.3select', type=str) # padgroupv1.22.1tp1select  singlepinpadv2.6impl singlepadv0.7select
parser.add_argument('--version', default='v2.17.3', type=str) #  v0.14_yl

parser.add_argument('--confidence', default='certain', type=str)
parser.add_argument('--output_type', default='dual2', type=str)  # gc_250520 bdg
parser.add_argument('--valdataset', default='bbtest', type=str, help='bbtest, bbtestmz, jiraissues, alltestval, newval, debug, cur_jiraissue, led_cross_pair')
parser.add_argument('--img_color', default='rgb', type=str)

parser.add_argument('--clean_train_data', default=False, type=bool) 
parser.add_argument('--clean_train_data_split_stage', default=2, type=int, help=f"数据太多,分批处理,阶段从0开始,仅在clean_train_data=True时有效")
parser.add_argument('--clean_train_data_split_length', default=10, type=int, help=f"数据太多,分批处理,每一批次处理多少个csv,仅在clean_train_data=True时有效")

parser.add_argument('--date', default='241022', type=str)   # 241022   debug-240723  body-241023_white  component-240913_white
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--threshold', default = 0.8, type=float, help='higher_threshold to check') # component - 0.7
parser.add_argument('--val_img_file_path', default=f'annotation_test_labels_pairs_region_output_type_flexible.csv', type=str)

parser.add_argument('--region', default=f'singlepinpad', type=str)   #padgroup
parser.add_argument('--Visualize', default=0, type=int)
parser.add_argument('--lut_p', default=0, type=float)

parser.add_argument('--down_csv', default=True, type=bool)
parser.add_argument('--only_down_csv', default=True, type=bool)
parser.add_argument('--down_csv_path', default='download_csv', type=str)

parser.add_argument('--filter_pairs_match_csv_path', default='singlepad_filter_pairs/match_csv_path', type=str)
parser.add_argument('--filter_pairs_defect_path', default='singlepad_filter_pairs/defect_path', type=str)

parser.add_argument('--show_low_thresh', default=False, type=bool)
parser.add_argument('--show_high_thresh', default=True, type=bool)
parser.add_argument('--wrong_predict_pair_save_path', default='./{region}_{label}_{img_color}_model_predict_{worr}_{version_folder}', type=str)
parser.add_argument('--if_split_val_ng_and_ok', default=False, type=bool)
parser.add_argument('--split_get_ng_or_ok', default='all', type=str, choices=['ok', 'ng', 'all'], help=f'当if_split_val_ng_and_ok打开的时候,\
                    split_get_ng_or_ok设置位ng和ok才会有意义')
parser.add_argument('--show_model_predict_wrong_or_right', default='wrong', type=str, choices=['right', 'wrong', 'all'],help = '可视化模型预测错误还是正确的结果')
parser.add_argument('--save_pkl', default=False, type=bool)
parser.add_argument('--reload_data', default=False, type=bool)
parser.add_argument('--light_device', default='all', type=str, choices=['2d', '3d', 'all'])
parser.add_argument('--ues_gradcam', default=False, type=bool)
parser.add_argument('--target_module', type=str, default='cnn_encoder', help = '要进行gradcam的层')
parser.add_argument('--target_layer', type=str, default='0', help = '要进行gradcam的层') # padgroup = 0, singlepinpad = 12, singlepad = 0, body=0
parser.add_argument('--cam_type', type=str, default='bos', help = '要进行gradcam的层')
parser.add_argument('--cam_target', type=str, default='insp', help = '对检测板还是金板计算gradcam') 
parser.add_argument('--calibration_T', default=1.0, type=float)

parser.add_argument('--cam_save_path', type=str, default='GradCAM_Results/{region}_{target_module}_{target_layer}_{cam_type}', 
                        help = '对检测板还是金板计算gradcam')   
parser.add_argument('--inference_model_mode', default='torch', type=str, choices=['torch', 'onnx_slim'])

args = parser.parse_args()

if args.clean_train_data:
    print(f"请确认clean_train_data_split_stage和valdataset已经更新: 输入(yes)确认")
    order = input()
    while order != 'yes':
        order = input()

    args.valdataset = args.valdataset + f"_clean_sateg{args.clean_train_data_split_stage}"

args.wrong_predict_pair_save_path = args.wrong_predict_pair_save_path.format(region = args.region,
                                                                             valdataset = args.valdataset,
                                                                             worr = args.show_model_predict_wrong_or_right,
                                                                             label = args.split_get_ng_or_ok,
                                                                             img_color = args.img_color,
                                                                             version_folder = args.version_folder)

args.cam_save_path = args.cam_save_path.format(
        region = args.region,
        target_module = args.target_module,
        target_layer = args.target_layer,
        cam_type = args.cam_type,
    )
cam_result_save_dir = os.path.join(args.wrong_predict_pair_save_path, args.valdataset, args.cam_save_path)
os.makedirs(cam_result_save_dir, exist_ok=True)

def split_data(val_image_pair_data, max_length=60000):
    """
    将数据分割成多个子集，每个子集最大长度为max_length，并保持索引连续
    
    参数:
        val_image_pair_data: 要分割的数据列表
        max_length: 每个子集的最大长度，默认为100
        
    返回:
        分割后的子集列表，每个子集都是一个字典，包含'data'和'start_index'
    """
    if len(val_image_pair_data) <= max_length:
        # return [{'data': val_image_pair_data, 'start_index': 0}]
        return [val_image_pair_data]

    subsets = []
    num_subsets = (len(val_image_pair_data) + max_length - 1) // max_length
    
    for i in range(num_subsets):
        start = i * max_length
        end = start + max_length
        subset = val_image_pair_data[start:end]
        subsets.append(subset)
    
    return subsets

def split_val_ng_and_ok(val_image_pair_path_list, get_ng_or_ok):
    val_image_pair_data_list = []
    ok_image_pair_data_list = []
    val_img_csv_dict = {}
    new_csv = {}  
    ng_nums= 0
    for path in tqdm(val_image_pair_path_list):
        csv_name = os.path.basename(path).split('.')[0]
        print(f"processing {csv_name}")

        df = pd.read_csv(path, index_col=0)
        if 'confidence' not in df.columns:
                df['confidence'] = 'certain'
        try:
            ng_df = df[df['binary_y']]
            ok_df = df[~df['binary_y']]
            ng_nums += len(ng_df)
            print(f"{csv_name} has {len(ng_df)} ng pairs")
            if 'ref_xy' not in ng_df.columns:
                if len(ng_df) > 0:
                    get_img_xy_from_img_name(ng_df)
                else:
                    ng_df['ref_xy'] = np.nan
                    ng_df['insp_xy'] = np.nan

                if len(ok_df) > 0:
                    get_img_xy_from_img_name(ok_df)
                else:
                    ok_df['ref_xy'] = np.nan
                    ok_df['insp_xy'] = np.nan

            if 'material_id' not in ng_df.columns:
                if len(ng_df) > 0:
                    get_material_id_from_img_name(ng_df)
                else:
                    ng_df['material_id'] = np.nan
                if len(ok_df) > 0:
                    get_material_id_from_img_name(ok_df)
                else:
                    ok_df['material_id'] = np.nan


            extracted_ng_data = ng_df[['ref_image', 'insp_image', 'ref_y', 'insp_y', 'ref_y_raw',
                                        'insp_y_raw', 'binary_y', 'material_id', 'ref_xy', 'insp_xy', 'confidence']]
            
            extracted_ok_data = ok_df[['ref_image', 'insp_image', 'ref_y', 'insp_y', 'ref_y_raw',
                                        'insp_y_raw', 'binary_y', 'material_id', 'ref_xy', 'insp_xy', 'confidence']]


            if 'defect_label' in ng_df.columns:
                extracted_ng_data['insp_defect_label'] = ng_df[['defect_label']]
                extracted_ok_data['insp_defect_label'] = ok_df[['defect_label']]

            else:
                if 'insp_defect_label' not in df.columns or df['insp_defect_label'].isna().any():
                    extracted_ng_data['insp_defect_label'] = ng_df[['singlepad_defect_label']]
                    extracted_ok_data['insp_defect_label'] = ok_df[['singlepad_defect_label']]
                else:
                    extracted_ng_data['insp_defect_label'] = ng_df[['insp_defect_label']]
                    extracted_ok_data['insp_defect_label'] = ok_df[['insp_defect_label']]
        except:
            raise ValueError('f{csv_name} has something wrong')
        extracted_ng_data['csv_name'] = csv_name
        extracted_ok_data['csv_name'] = csv_name

        if get_ng_or_ok == "ng":
            val_image_pair_data_list.append(extracted_ng_data)
            val_img_csv_dict[csv_name] = extracted_ng_data
        elif get_ng_or_ok == "ok":
            val_image_pair_data_list.append(extracted_ok_data)
            val_img_csv_dict[csv_name] = extracted_ok_data
        else:
            raise ValueError(f'get_ng_or_ok只能是ok或者ng')

        ok_image_pair_data_list.append(extracted_ok_data)
        new_csv[csv_name] = []
    print(f"total ng pairs: {ng_nums}")

    return val_image_pair_data_list, ok_image_pair_data_list, val_img_csv_dict, new_csv

def get_data_in_str_list(str_list):
    # 使用正则表达式提取数字（包括整数和浮点数）
    numbers = []
    for item in str_list:
        # 使用正则表达式匹配数字（整数和浮点数）
        matches = re.findall(r'-?\d+\.?\d*', item)
        # 将匹配的字符串转换为浮点数并添加到列表
        numbers.extend(float(match) for match in matches)

    return numbers

def get_img_xy_from_img_name(df):
    for index, infos in df.iterrows():
        ref_image_name = infos['ref_image'].split('/')[-1]
        insp_image_name = infos['insp_image'].split('/')[-1]

        ref_infos = ref_image_name.split('sp')[-1].split('_')
        insp_infos = insp_image_name.split('sp')[-1].split('_')

        candidate_ref_xy = get_data_in_str_list(ref_infos)
        candidate_insp_xy =  get_data_in_str_list(insp_infos)
        if len(candidate_ref_xy) == 2:
            ref_xy = str(candidate_ref_xy[0]) + '_' + str(candidate_ref_xy[1])
            insp_xy = str(candidate_insp_xy[0]) + '_' + str(candidate_insp_xy[1])
        else:
            ref_xy = 0
            insp_xy = 0

        # 将计算得到的值添加到 DataFrame 的新列中
        df.at[index, 'ref_xy'] = ref_xy
        df.at[index, 'insp_xy'] = insp_xy

def get_material_id_from_img_name(df):
    for index, infos in df.iterrows():
        material_id = infos['ref_image'].split('/')[1]
        # 将计算得到的值添加到 DataFrame 的新列中
        df.at[index, 'material_id'] = material_id

def visualize_pair(ref_img, insp_img, insp_label, binary_label, certainty, fig_path):
    ref_image_show = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    insp_image_show = cv2.cvtColor(insp_img, cv2.COLOR_BGR2RGB)

    # plot reference image against inspected image
    if ref_img.shape[1] / ref_img.shape[0] > 2.5:
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    else:
        fig, axes = plt.subplots(1, 2, figsize=(6, 6))

    axes[0].imshow(ref_image_show, cmap='gray')
    axes[0].set_title(f'Ref Binary NG = ({binary_label})')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(insp_image_show, cmap='gray')
    axes[1].set_title(f'Insp Mclass = {insp_label} \n ({certainty})')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.savefig(fig_path)
    plt.close()
    return
random.seed(42)
np.random.seed(42)
##################################################################################################################

img_color = args.img_color
if args.if_split_val_ng_and_ok:
    assert args.split_get_ng_or_ok != 'all', 'if_split_val_ng_and_ok为True时， split_get_ng_or_ok不可以为all'
else:
    assert args.split_get_ng_or_ok == 'all', 'if_split_val_ng_and_ok为False时， split_get_ng_or_ok必须为all'

predict_wrong_save_path = os.path.join(args.wrong_predict_pair_save_path, args.valdataset)
os.makedirs(predict_wrong_save_path, exist_ok=True)

compare_trt = False
# inference_model_mode = 'onnx_slim'
# inference_model_mode = 'torch'
inference_model_mode = args.inference_model_mode
print(f"inference_model_mode: {inference_model_mode}")

seed = 42

# ckp_folder = '/mnt/pvc-nfs-dynamic/robinru/model_ckps/aoi-sdkv0.10.0'
# ckp_folder = f'./models/checkpoints/{args.version_folder}'
ckp_folder = f'./models/checkpoints/{args.region}/{args.version}/{args.version_folder}'

region_list = args.region.split(';')
if 'component' in region_list or 'body' in region_list:
    image_folder =  '/mnt/dataset/xianjianming/data_clean_white/' #'/mnt/ssd/classification/data/data_clean_white/'
else:
    image_folder = '/mnt/dataset/xianjianming/data_clean/' #'/mnt/ssd/classification/data/data_clean/'

if args.valdataset == 'debug':
    image_folder = '/mnt/dataset/xianjianming/debug/' #'/mnt/ssd/classification/data/data_clean/'

annotation_folder = os.path.join(image_folder, 'merged_annotation', args.date)

# image normalization mean and std
gpu_exists = True
print_unique = False

# region_list = ['singlepad', 'singlepinpad', 'padgroup']
lut_p = args.lut_p
compute_ensemble = False
batch_size = 256
version_folder = args.version_folder
label_confidence = args.confidence
higher_threshold = args.threshold
gamma = 2
date = args.date
output_type = args.output_type
gpu_id = args.gpu
val_img_file_path = args.val_img_file_path.replace('output_type', output_type)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
image_saving_phase = 'train'
print(output_type, version_folder, gpu_id, label_confidence, inference_model_mode)
run_time = []

compare_version_res_folder = f'./results/{args.region}/{version_folder}_{date}_filter/{args.valdataset}_T{args.calibration_T}'
os.makedirs(compare_version_res_folder, exist_ok=True)

compare_versions = {'backbone': [], 'region': [], 'version_name': [], 'n_params': [], 'infer_time': [],
                    'bi_acc': [], 'mclass_acc': [], 'epoch': []}

compare_versions.update({f'bi_ng_{metric}': [] for metric in ['precision', 'recall', 'f1_score', 'omission_rate']})
compare_versions.update({f'mclass_ng_{metric}': [] for metric in ['precision', 'recall', 'f1_score', 'omission_rate']})

compare_versions.update({f'bi_ok_{metric}': [] for metric in ['precision', 'recall', 'f1_score', 'omission_rate']})
compare_versions.update({f'mclass_ok_{metric}': [] for metric in ['precision', 'recall', 'f1_score', 'omission_rate']})

all_store_df_list = []
for region in region_list:

    if region == 'pins_part' or region == 'padgroup':
        if '_' in version_folder:
            rs_img_size_w = int(version_folder.split('_')[1])
            rs_img_size_h = int(version_folder.split('_')[2])
#             img_color = version_folder.split('_')[3]
            # img_color = 'rgb'
        else:
            rs_img_size_w, rs_img_size_h = 128, 128
            # img_color = 'rgb'
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'
        if args.clean_train_data:
            annotation_filename, val_annotation_filename, test_annotation_filenames, aug_train_pair_data_filenames, aug_val_pair_data_filenames = \
                get_train_csv(annotation_folder, 'padgroup', version_folder,special_data = 'filter_train_data')
            val_image_pair_path_list = aug_train_pair_data_filenames + aug_val_pair_data_filenames + test_annotation_filenames
            if annotation_filename is not None:
                val_image_pair_path_list.append(annotation_filename)

            if val_annotation_filename is not None:
                val_image_pair_path_list.append(annotation_filename)
            
        else:
            val_image_pair_path_list = get_test_all_csvs(image_folder, date,  'padgroup', args.valdataset, img_color)

    elif region == 'single_pin' or region == 'singlepinpad':
        anno_folder = os.path.join(image_folder, 'merged_annotation', date)
        rs_img_size_w = 128
        rs_img_size_h = 32
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'
        if args.clean_train_data:
            annotation_filename, val_annotation_filename, test_annotation_filenames, aug_train_pair_data_filenames, aug_val_pair_data_filenames = \
                get_train_csv(annotation_folder, 'singlepinpad', version_folder,special_data = 'filter_train_data')
            val_image_pair_path_list = aug_train_pair_data_filenames + aug_val_pair_data_filenames + test_annotation_filenames
            if annotation_filename is not None:
                val_image_pair_path_list.append(annotation_filename)

            if val_annotation_filename is not None:
                val_image_pair_path_list.append(annotation_filename)
            
        else:
            val_image_pair_path_list = get_test_all_csvs(image_folder, date,  'singlepinpad', args.valdataset, img_color)

    elif region == 'singlepad':
        if args.light_device == '2d' or args.light_device == '3d':
            tmpt_image_folder = image_folder
            tmpt_date = date
            image_folder = '/mnt/pvc-nfs-dynamic/robinru/data/classification/data_clean'
            date = '241209'
        anno_folder = os.path.join(image_folder, 'merged_annotation', date)
        rs_img_size_w = 64
        rs_img_size_h = 64
                                
        print(f"rs_img_size_w = {rs_img_size_w} \n rs_img_size_h = {rs_img_size_h}")
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'

        if args.clean_train_data:
            annotation_filename, val_annotation_filename, test_annotation_filenames, aug_train_pair_data_filenames, aug_val_pair_data_filenames = \
                get_train_csv(annotation_folder, 'singlepad', version_folder,special_data = 'filter_train_data')
            val_image_pair_path_list = aug_train_pair_data_filenames + aug_val_pair_data_filenames + test_annotation_filenames
            if annotation_filename is not None:
                val_image_pair_path_list.append(annotation_filename)

            if val_annotation_filename is not None:
                val_image_pair_path_list.append(annotation_filename)
            
        else:
            val_image_pair_path_list = get_test_all_csvs(image_folder, date,  'singlepad', args.valdataset, img_color)
        
    elif region == 'component':

        anno_folder = os.path.join(image_folder, 'merged_annotation', date)
        rs_img_size_w = 224
        rs_img_size_h = 224
        rsstr = f'rs{rs_img_size_w}'
        val_image_pair_path_list = get_test_all_csvs(image_folder, date,  'component', args.valdataset, img_color)

    elif region == 'body':
        anno_folder = os.path.join(image_folder, 'merged_annotation', date)
        candidate_size = args.version_folder.split('r')[-1]
        if candidate_size.isdigit():
            rs_img_size_w = int(candidate_size)
            rs_img_size_h = int(candidate_size)
        else:
            rs_img_size_w = 224
            rs_img_size_h = 224
        rsstr = f'rs{rs_img_size_w}'
        if args.clean_train_data:
            annotation_filename, val_annotation_filename, test_annotation_filenames, aug_train_pair_data_filenames, aug_val_pair_data_filenames = \
                get_train_csv(annotation_folder, 'body', version_folder, special_data = 'filter_train_data')
            val_image_pair_path_list = aug_train_pair_data_filenames + aug_val_pair_data_filenames + test_annotation_filenames
            if annotation_filename is not None:
                val_image_pair_path_list.append(annotation_filename)

            if val_annotation_filename is not None:
                val_image_pair_path_list.append(annotation_filename)
            
        else:
            val_image_pair_path_list = get_test_all_csvs(image_folder, date,  'body', args.valdataset, img_color)


    tmp_list= []
    [tmp_list.append(x) for x in val_image_pair_path_list  if x not in tmp_list]
    val_image_pair_path_list = deepcopy(tmp_list)
    print(args.valdataset, val_image_pair_path_list)


    if args.down_csv:
        os.makedirs(args.down_csv_path, exist_ok=True)
        for csv_path in val_image_pair_path_list:
            csv_name = os.path.basename(csv_path)
            down_csv_path = os.path.join(f'./{args.down_csv_path}_{args.region}_{args.valdataset}' , csv_name)
            os.system(f"sudo cp {csv_path} {down_csv_path}")
        
        if args.only_down_csv:
            print("只下载 CSV 文件，程序终止。")
            sys.exit(0)

    if args.clean_train_data:
        start = args.clean_train_data_split_stage * args.clean_train_data_split_length
        end = (args.clean_train_data_split_stage + 1) * args.clean_train_data_split_length
        if end > len(val_image_pair_path_list):
            print(f"已全部处理完")
            val_image_pair_path_list = val_image_pair_path_list[start:]
        else:
            val_image_pair_path_list = val_image_pair_path_list[start:end]

    if inference_model_mode == 'torch':
        version_name_list = [f.split('/')[-1] for f in glob.glob(os.path.join(ckp_folder, f'*pth*'))  #os.path.join(ckp_folder, version_folder, '*')
                             if region in f.split('/')[-1] and 'pth.tar' in f.split('/')[-1] and output_type in
                             f.split('/')[-1] and (rsstr in f)]
    elif 'onnx' in inference_model_mode:
        version_name_list = [f.split('/')[-1] for f in glob.glob(os.path.join(ckp_folder, f'*onnx'))  # version_folder, '*' 
                             if region in f.split('/')[-1] and 'slim.onnx' in f.split('/')[-1] and output_type in
                             f.split('/')[-1] and (rsstr in f)]
    print(version_name_list)
    print(f'# ---- {region.upper()} RESULTS ---- #')
    # specify region of interest and defect types

    # decode the defect id back to defect label

    if region == 'component':
        if 'fly' in version_folder:
            defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11, 'fly': 12}
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4, 'tombstone': 5, 'others': 6, 'fly': 7}
            defect_code_slim = {'ok': 0, 'missing': 1,'tombstone': 8, 'others': 11,  'fly': 12}
#             defect_code_slim = {'ok': 0, 'missing': 1, 'wrong': 3,'tombstone': 8, 'others': 11}
            attention_list = None
            if 'flycbam1' in version_folder:
                attention_list_str = version_folder.split('flycbam')[-1]
                attention_list = [int(attention_list_str)]

        else:
            defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4, 'tombstone': 5, 'others': 6}
            defect_code_slim = {'ok': 0, 'missing': 1,'tombstone': 8, 'others': 11}

    elif region == 'body' or region == 'bodyl':

        if 'slim' in version_folder:
#             defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
            defect_code = {'ok': 0, 'wrong': 3}  # v1.32
            defect_code_slim = defect_code
            defect_sdk_decode = {'ok': 2, 'wrong': 4}
        else:
            defect_code = {'ok': 0, 'missing': 1, 'wrong': 3}  # v1.32
            defect_code_slim = defect_code
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4}
#             defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
#             defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4, 'tombstone': 5, 'others': 6}
#             defect_code_slim = {'ok': 0, 'missing': 1, 'wrong': 3}            

    elif region == 'single_pin' or region == 'singlepinpad':
#         if 'singlepinpadold' in version_folder or 'singlepinpadv1.9.x' in version_folder:
#             defect_code = {'ok': 0, 'undersolder': 4, 'oversolder': 5, 'pseudosolder': 6}
#         else:
        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
            
        defect_code_slim = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_sdk_decode = {'ok': 2, 'undersolder': 3, 'pseudosolder': 4}

    elif region == 'singlepad':
        defect_code_slim = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_sdk_decode = {'ok': 2, 'undersolder': 3, 'pseudosolder': 4}

    elif region == 'pins_part' or region == 'padgroup':
        defect_code_slim = {'ok': 0, 'solder_shortage': 7}
        if rsstr == f'rs224112':
            defect_code = {'ok': 0, 'missing': 1, 'undersolder': 4, 'pseudosolder': 6, 'solder_shortage': 7}
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'undersolder': 4, 'pseudosolder': 5, 'solder_shortage': 6}  # v1.32
        elif 'nonslim' in version_folder:
            defect_code = {'ok': 0, 'missing': 1, 'solder_shortage': 7}
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'solder_shortage': 4}  # v1.32
        else:
            defect_code = {'ok': 0, 'solder_shortage': 7}
            defect_sdk_decode = {'ok': 2, 'solder_shortage': 3}  # v1.32

    n_class = len(defect_code)
    # defect_decode = {i: v for i, v in enumerate(defect_list)}
########################################仅用于对比fly和nofly模型####################################################################
    # defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
    # defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4, 'tombstone': 5, 'others': 6}
    # defect_code_slim = {'ok': 0, 'missing': 1,'tombstone': 8, 'others': 11}
############################################################################################################
    defect_full_decode = {v: k for k, v in defect_sdk_decode.items()}
    defect_decode = {i: k for i, k in enumerate(defect_code.keys())}
    defect_convert = {v: i for i, v in enumerate(defect_code.values())}
    defect_decode.update({-1: 'ng'})
    # load test images  
    # 创建一个空的 DataFrame 用于存储合并的数据
    combined_ng_df = pd.DataFrame() 
    combined_ok_df = pd.DataFrame() 
    if args.if_split_val_ng_and_ok:
        val_image_pair_data_list, ok_image_pair_data_list, val_img_csv_dict, new_csv = split_val_ng_and_ok(val_image_pair_path_list, args.split_get_ng_or_ok)
        ok_image_pair_data_raw = pd.concat(ok_image_pair_data_list).reset_index(drop=True)
        ok_image_pair_data_raw = ok_image_pair_data_raw.drop_duplicates(subset=['ref_image', 'insp_image'])
        ok_image_pair_data_raw.to_csv(os.path.join(predict_wrong_save_path, 'ori_ok.csv'))
    else:
        val_image_pair_data_list = []
        val_img_csv_dict = {}
        new_csv = {} 
        # val_image_pair_data_list = [pd.read_csv(path, index_col=0) for path in val_image_pair_path_list]
        for path in val_image_pair_path_list:
            if path is None:
                continue
            csv_name = os.path.basename(path).split('.')[0]
            csv_df = pd.read_csv(path)
            csv_df['csv_name'] = csv_name
            if 'confidence' not in csv_df.columns:
                csv_df['confidence'] = 'certain'

            val_image_pair_data_list.append(csv_df)
            val_img_csv_dict[csv_name] = csv_df

            new_csv[csv_name] = []

    # val_image_pair_data_list = [pd.read_csv(path, index_col=0) for path in val_image_pair_path_list]
    pkl_infos_keys = [
        'val_image_pair_res.pkl',
        'binary_label_list.pkl',
        'ref_image_list.pkl',
        'insp_image_list.pkl',
        'ref_label_list.pkl',
        'insp_label_list.pkl',
        'ref_image_name_list.pkl',
        'insp_image_name_list.pkl',
        'counter.pkl',
        'ref_image_batches.pkl',
        'insp_image_batches.pkl',
        'ref_image_sub_dir_path.pkl',
        'insp_image_sub_dir_path.pkl',
        'csv_names.pkl',
        'n_batches.pkl',
        'batch_idices_list.pkl'
        ]
    
    if args.reload_data:
        loaded_data = {}  
        for pkl_name in pkl_infos_keys:
            pkl_path = './' + pkl_name
            # if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as file:
                    loaded_data[pkl_name] = pickle.load(file)
                    print(f"成功加载 {pkl_name}")
            except:
                raise ValueError(f'{pkl_path}加载失败')

        val_image_pair_res = loaded_data['val_image_pair_res.pkl']
        binary_label_list = loaded_data['binary_label_list.pkl']
        ref_image_list = loaded_data['ref_image_list.pkl']
        insp_image_list = loaded_data['insp_image_list.pkl']
        ref_label_list = loaded_data['ref_label_list.pkl']
        insp_label_list = loaded_data['insp_label_list.pkl']
        ref_image_name_list = loaded_data['ref_image_name_list.pkl']
        insp_image_name_list = loaded_data['insp_image_name_list.pkl']
        counter = loaded_data['counter.pkl']
        ref_image_batches = loaded_data['ref_image_batches.pkl']
        insp_image_batches = loaded_data['insp_image_batches.pkl']
        ref_image_sub_dir_path = loaded_data['ref_image_sub_dir_path.pkl']
        insp_image_sub_dir_path = loaded_data['insp_image_sub_dir_path.pkl']
        csv_names = loaded_data['csv_names.pkl']
        n_batches = loaded_data['n_batches.pkl']
        batch_idices_list = loaded_data['batch_idices_list.pkl']

    else:
        val_image_pair_data_raw = pd.concat(val_image_pair_data_list)  # 
        val_image_pair_data_raw = val_image_pair_data_raw.drop_duplicates(subset=['ref_image', 'insp_image']).reset_index(drop=True)
        
        if 'defect_dir' in val_image_pair_data_raw.columns:
            val_image_pair_data_raw = val_image_pair_data_raw[val_image_pair_data_raw['defect_dir'] != 'WORD_UNPAIR']

        if args.valdataset == 'train':
            val_image_pair_data_raw = val_image_pair_data_raw.drop_duplicates().reset_index(drop=True)
            val_image_pair_data_raw['confidence'] = 'certain'
            print(len(val_image_pair_data_raw))
            val_image_pair_data = val_image_pair_data_raw.copy()

        else:
            if region == 'body' and 'slim' in version_folder:
                val_image_pair_data_raw['insp_y_raw_old'] = list(val_image_pair_data_raw['insp_y_raw'])
                val_image_pair_data_raw['insp_y_raw'] = [3 if y == 1 else y for y in val_image_pair_data_raw['insp_y_raw_old']]  

            if region == 'pins_part':
                val_image_pair_data = val_image_pair_data_raw[
                    [y in list(defect_code_slim.values()) for y in val_image_pair_data_raw['insp_y_raw']]].reset_index(
                    drop=True)
            else:
                val_image_pair_data = val_image_pair_data_raw[
                    [y in list(defect_code_slim.values()) for y in val_image_pair_data_raw['insp_y_raw']]]
            val_image_pair_data['insp_y'] = [defect_convert[yraw] for yraw in val_image_pair_data['insp_y_raw']]

        if label_confidence == 'certain':
            val_image_pair_data = val_image_pair_data[
                (val_image_pair_data['confidence'] == 'certain') | (
                        val_image_pair_data['confidence'] == 'unchecked')].copy().reset_index(drop=True)
        elif label_confidence == 'uncertain':
            val_image_pair_data = val_image_pair_data[
                (val_image_pair_data['confidence'] == 'uncertain') ].copy().reset_index(drop=True)
        elif label_confidence == 'all':
            val_image_pair_data = val_image_pair_data[ 
                (val_image_pair_data['confidence'] != 'BAD_PAIR')].copy().reset_index(drop=True)
            
        if args.light_device == '2d':
            val_image_pair_data = val_image_pair_data[
                (val_image_pair_data['feature_set_name'] != 'default3D') &
                (val_image_pair_data['feature_set_name'] != 'temp3D')].copy().reset_index(drop=True)
            image_folder = tmpt_image_folder
            date = tmpt_date
        elif args.light_device == '3d':
            val_image_pair_data = val_image_pair_data[
                (val_image_pair_data['feature_set_name'] == 'default3D') |
                (val_image_pair_data['feature_set_name'] == 'temp3D')].copy().reset_index(drop=True)
            image_folder = tmpt_image_folder
            date = tmpt_date
            
        # val_image_pair_data = val_image_pair_data.reset_index(drop=True)
        print(f"去重后的{args.split_get_ng_or_ok}数据集数量：{len(val_image_pair_data)}")

        val_image_pair_res = val_image_pair_data[['ref_image', 'insp_image']].copy()
        binary_label_list = val_image_pair_data['binary_y'].tolist()
        ref_image_list = []
        insp_image_list = []
        ref_label_list = []
        insp_label_list = []
        ref_image_name_list = []
        insp_image_name_list = []
        counter = 0
        ref_image_batches = []
        insp_image_batches = []
        ref_image_sub_dir_path = []
        insp_image_sub_dir_path = []
        csv_names = []
        n_batches = int(len(val_image_pair_data) / batch_size)

        if len(val_image_pair_data) < batch_size:
            batch_idices_list = [range(len(val_image_pair_data))]
            n_batches = 1
        else:
            batch_idices_list = split(list(val_image_pair_data.index), n_batches)
            
        img_folder = '/mnt/pvc-nfs-dynamic/xianjianming/data/'
        transform = transforms.Compose([ # 非同步的
            LUT_VAL(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'), lut_p=lut_p),

        ])  

        for batch_indices in batch_idices_list:

            ref_batch = []
            insp_batch = []
            for i in batch_indices:

                if counter % 1000 == 0:
                    print(f'{counter}')
                counter += 1
                val_image_pair_i = val_image_pair_data.iloc[i]

                ref_image_path = os.path.join(image_folder, val_image_pair_i['ref_image'])
                insp_image_path = os.path.join(image_folder, val_image_pair_i['insp_image'])

                ref_image_sub_dir_path.append('/'.join(val_image_pair_i['ref_image'].split('/')[:-1]))
                insp_image_sub_dir_path.append('/'.join(val_image_pair_i['insp_image'].split('/')[:-1]))

                ref_image_name = val_image_pair_i['ref_image'].split('/')[-1]
                insp_image_name = val_image_pair_i['insp_image'].split('/')[-1]
                csv_names.append(val_image_pair_i['csv_name'])
                # scale, normalize and resize test images
                ref_image = TransformImage(img_path=ref_image_path, rs_img_size_h=rs_img_size_h,
                                        rs_img_size_w=rs_img_size_w, transform=transform,
                                        ).transform()  # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p
                insp_image = TransformImage(img_path=insp_image_path, rs_img_size_h=rs_img_size_h,
                                            rs_img_size_w=rs_img_size_w, transform=transform,
                                            ).transform() # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p

                # get ground truth labels
                ref_label = val_image_pair_i['ref_y']
                insp_label = val_image_pair_i['insp_y']
                ref_batch.append(ref_image)
                insp_batch.append(insp_image)

                ref_label_list.append(ref_label)
                insp_label_list.append(insp_label)
                ref_image_name_list.append(ref_image_name)
                insp_image_name_list.append(insp_image_name)

            ref_image_batches.append(np.concatenate(ref_batch, axis=0))
            insp_image_batches.append(np.concatenate(insp_batch, axis=0))

        pkl_infos = [
            val_image_pair_res,
            binary_label_list,
            ref_image_list,
            insp_image_list,
            ref_label_list,
            insp_label_list,
            ref_image_name_list,
            insp_image_name_list,
            counter,
            ref_image_batches,
            insp_image_batches,
            ref_image_sub_dir_path,
            insp_image_sub_dir_path,
            csv_names,
            n_batches,
            batch_idices_list
        ]
        # 将列表保存为 pkl 文件
        if args.save_pkl:
            for index, pkl_name in enumerate(pkl_infos_keys):
                with open(pkl_name, 'wb') as file:
                    pickle.dump(pkl_infos[index], file)
        
    n_batches = np.max([int(len(ref_image_list) / batch_size), 1])

    binary_labels = np.array(binary_label_list).reshape([-1, 1])
    ref_labels = np.array(ref_label_list).reshape([-1, 1])
    insp_labels = np.array(insp_label_list).reshape([-1, 1])

    for version_name in version_name_list:
        backbone_arch = version_name.split('rs')[0].split(region)[-1]
        if 'nb' not in version_name:
            n_units = [128, 128]
        else:
            if 'top' in version_name:
                n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                        version_name.split('f16')[-1].split('top')[0].split('nb')[-1].split('nm')]
            elif 'last' in version_name:
                n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                        version_name.split('f16')[-1].split('last')[0].split('nb')[-1].split('nm')]
            else:
                n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                           version_name.split('f16')[-1].split('last')[0].split('nb')[-1].split('nm')]
                # n_units = [128, 128]
        model_outputs = {'binary_outputs': {}, 'mclass_outputs': {}, 'binary_labels': {}, 'mclass_labels': {},
                         'insp_image_names': {}}
        trt_model_outputs = {'binary_outputs': {}, 'mclass_outputs': {}, 'binary_labels': {}, 'mclass_labels': {},
                             'insp_image_names': {}}
        if compare_trt:
            trt_results = pd.read_csv(os.path.join(f'./results/{version_folder}/', f'{region}_0615_defect_result.csv'),
                                      index_col=False)
            trt_results.columns = ['ref_image', 'insp_image', 'trt_binary_class', 'trt_binary_p', 'trt_mclass_raw',
                                   'trt_mclass_p']
            trt_encode = {k: i for i, k in enumerate(defect_code.keys())}
            trt_results['trt_mclass'] = [trt_encode[defect_full_decode[t]] for t in trt_results['trt_mclass_raw']]

        if inference_model_mode == 'torch':

            torch_ckp_path = os.path.join(ckp_folder, #version_folder,
                                          f'{version_name}')

            # fix random seed for reproducibility
            np.random.seed(seed)
            torch.manual_seed(seed)

            if 'CLL' in version_name:
                output_type = "dual2CLL"
            elif 'CL' in version_name:
                output_type = "dual2CL"
            else:
                output_type = "dual2"

            # define model and load check point
            # if 'resnetsp' in backbone_arch:
            #     from models.MPB3 import MPB3net
            if 'cbam' in backbone_arch and 'CL' in version_name:
                print(backbone_arch)
                if region == 'body' or region == 'component':
                    attention_list_str = backbone_arch.split('cbam')[-1]
                    if len(attention_list_str.split('cbam')[-1]) == 0:
                        attention_list = None
                    else:
                        attention_list = [int(attention_list_str)]
                
                from models.MPB3_attn_ConstLearning import MPB3net
                print(attention_list)  

                model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                                  output_form=output_type, attention_list=attention_list)
            elif 'cbam' in backbone_arch and 'CL' not in version_name:
                print(backbone_arch)
                if region == 'body' or region == 'component':
                    attention_list_str = backbone_arch.split('cbam')[-1]
                    if len(attention_list_str.split('cbam')[-1]) == 0:
                        attention_list = None
                    else:
                        attention_list = [int(attention_list_str)]
                
                from models.MPB3_attention import MPB3net
                print(attention_list)
                model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                                  output_form=output_type, attention_list=attention_list)
            elif 'CL' in version_name:
                from models.MPB3_ConstLearning import MPB3net

                model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                                output_form=output_type)

            else:
                from models.MPB3 import MPB3net

                model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                                output_form=output_type)


            try:
                print(f'=> Loading checkpoint from {torch_ckp_path}')
                checkpoint = torch.load(torch_ckp_path)
                ckp_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                print(f'=> Loaded checkpoint {torch_ckp_path} {ckp_epoch}')
                
            except:
                print(torch_ckp_path, 'No checkpoint found')
                continue
                # assert False, 'No checkpoint found'

            # switch to evaluate mode
            model.eval()
            predicted_defect_class_cC = {}
            predicted_binary_class_cC = {}
            predicted_defect_confidence_cC = {}
            predicted_binary_confidence_cC = {}

            if gpu_exists:
                model.cuda()

            n_params = sum([p.numel() for p in model.parameters()])
            # compute ouputs
            output_bos_list = []
            output_bom_list = []
            for img1, img2 in zip(ref_image_batches, insp_image_batches):

                time_start = time.time()
                if gpu_exists:
                    img1, img2 = torch.FloatTensor(img1).cuda(), torch.FloatTensor(img2).cuda()
                else:
                    img1, img2 = torch.FloatTensor(img1), torch.FloatTensor(img2)

                # compute loss and accuracy
                time_start = time.time()

                with torch.no_grad():
                    if 'CL' in version_name:
                        output_bos, output_bom, _, _ = model(img1, img2)
                    else:
                        output_bos, output_bom = model(img1, img2)

                time_end = time.time()
                batch_run_time = time_end - time_start
                batch_size = img1.size(0)

                # update record
                output_bos_list.append(output_bos)
                output_bom_list.append(output_bom)
                run_time.append(batch_run_time)

            print(
                f'avg: {np.mean(run_time[1:]) * 1000}ms, batch_size: {batch_size}, n_params={n_params / (1024 * 1024)}M')
            output_bos_all = torch.cat(output_bos_list, dim=0)
            output_bom_all = torch.cat(output_bom_list, dim=0)
            output_bos_np = output_bos_all.detach().cpu().numpy()
            output_bom_np = output_bom_all.detach().cpu().numpy()

            compare_versions['backbone'].append(backbone_arch)
            compare_versions['version_name'].append(version_name)
            compare_versions['region'].append(region)
            compare_versions['n_params'].append(n_params / (1024 * 1024))
            compare_versions['infer_time'].append(np.mean(run_time[1:]) * 1000)
            compare_versions['epoch'].append(ckp_epoch)
        elif 'onnx' in inference_model_mode:
            import onnx
            import onnxruntime as ort
            ckp_epoch = None
            n_params = 0
            if inference_model_mode == 'onnx':
                print('run onnx')
                # 'componentmobilenetv3largers224s42c10val0.0_ckp_bestv1.720.05fp16slim'
                onnx_ckp_path = os.path.join(ckp_folder,    # version_folder,
                                             f'{version_name}')
            else:
                print(f'run onnx slim {version_name}')

                onnx_ckp_path = os.path.join(ckp_folder,  # version_folder,
                                             f'{version_name}')

            # verify the onnx model using ONNX library
            onnx_model = onnx.load(onnx_ckp_path)
            onnx.checker.check_model(onnx_model)

            # run the onnx model with one of the runtimes that support ONNX
            ort_session = ort.InferenceSession(onnx_ckp_path, providers=['CUDAExecutionProvider'])
            output_bos_np_list, output_bom_np_list = [], []
            for ref_image_batch, insp_image_batch in zip(ref_image_batches, insp_image_batches):
                # ref_image_all = np.concatenate(ref_image_list
                #                , axis=0)
                # insp_image_all = np.concatenate(insp_image_list
                #                , axis=0)
                onnx_inputs = {ort_session.get_inputs()[i].name: image_batch for i, image_batch in
                               enumerate([ref_image_batch, insp_image_batch])}
                time_start = time.time()
                output_bos_np_batch, output_bom_np_batch = ort_session.run(None, onnx_inputs)
                output_bos_np_list.append(output_bos_np_batch)
                output_bom_np_list.append(output_bom_np_batch)
                time_end = time.time()
                batch_run_time = time_end - time_start
            print(f'avg: {batch_run_time * 1000}ms, batch_size: {batch_size}')

            # output_bos_all = torch.cat(output_bos_np_list, dim=0)
            # output_bom_all = torch.cat(output_bom_np_list, dim=0)
            # output_bos_np = output_bos_all.detach().cpu().numpy()
            # output_bom_np = output_bom_all.detach().cpu().numpy()

            compare_versions['backbone'].append(backbone_arch)
            compare_versions['version_name'].append(version_name)
            compare_versions['region'].append(region)
            compare_versions['n_params'].append(n_params / (1024 * 1024))
            compare_versions['infer_time'].append(np.mean(run_time[1:]) * 1000)
            compare_versions['epoch'].append(ckp_epoch)

            output_bos_np = np.vstack(output_bos_np_list)
            output_bom_np = np.vstack(output_bom_np_list)

        # apply softmax
        output_bos_np_softmax = softmax(output_bos_np, axis=1)
        output_bom_np_softmax = softmax(output_bom_np, axis=1)

        model_outputs['binary_outputs'][region] = output_bos_np_softmax
        model_outputs['mclass_outputs'][region] = output_bom_np_softmax
        model_outputs['binary_labels'][region] = binary_labels
        model_outputs['mclass_labels'][region] = insp_labels
        model_outputs['insp_image_names'][region] = insp_image_name_list

        val_image_pair_res[f'{inference_model_mode}_binary_class'] = np.argmax(output_bos_np_softmax, axis=1)
        val_image_pair_res[f'{inference_model_mode}_binary_p'] = output_bos_np_softmax[:, 1]
        val_image_pair_res[f'{inference_model_mode}_mclass'] = np.argmax(output_bom_np_softmax, axis=1)
        val_image_pair_res[f'{inference_model_mode}_mclass_p'] = np.max(output_bom_np_softmax, axis=1)
        val_image_pair_res['true_mclass'] = insp_label_list
        binary_labels = binary_labels.astype(int)
        binary_class = np.argmax(output_bos_np_softmax, axis=1).astype(np.int8)
        p_binary_class = np.max(output_bos_np_softmax, axis=1)

        exact_defect_class = np.argmax(output_bom_np_softmax, axis=1).astype(np.int8)
        p_defect_class = np.max(output_bom_np_softmax, axis=1)

        if args.show_low_thresh:
            for index, img_names  in enumerate(zip(ref_image_name_list, insp_image_name_list)):
                binary_label = binary_labels[index]
                insp_label = insp_labels[index]
                csv_name_path = csv_names[index]
                csv_df = val_img_csv_dict[csv_name_path]
                if int(insp_label) < len(defect_decode) - 1:
                    mclass_defect = defect_decode[int(insp_label)]
                else:
                    raise 

                predict_binary = binary_class[index]
                predict_binary_score = p_binary_class[index]
                predict_mclass = exact_defect_class[index]
                predict_multi_score = p_defect_class[index]
                if predict_mclass < len(defect_decode) - 1:
                    predict_mclass_defect = defect_decode[predict_mclass]
                else:
                    raise
                ref_image_sub_dir_path_ = ref_image_sub_dir_path[index]
                insp_image_sub_dir_path_ = insp_image_sub_dir_path[index]
                ref_image_name, insp_img_name = img_names
                
                ref_img_relative_path = os.path.join(ref_image_sub_dir_path_, ref_image_name)
                insp_img_relative_path = os.path.join(insp_image_sub_dir_path_, insp_img_name)

                csv_df_raws = csv_df[((csv_df['ref_image'] == ref_img_relative_path) & 
                                    (csv_df['insp_image'] == insp_img_relative_path))]
                if predict_mclass_defect == mclass_defect:
                    csv_df_raws['model_predict_right'] = True
                    new_csv[csv_name_path].append(csv_df_raws)
                    continue
                else:
                    csv_df_raws['model_predict_right'] = False
                    new_csv[csv_name_path].append(csv_df_raws)
                    
                    ref_img_path = os.path.join(image_folder, ref_image_sub_dir_path_, ref_image_name)
                    insp_img_path = os.path.join(image_folder, insp_image_sub_dir_path_, insp_img_name)

                    ref_img = cv2.imread(ref_img_path)
                    insp_img = cv2.imread(insp_img_path)

                    if '#' in ref_image_name:
                        ref_img_name = ref_image_name.split('#')[0]
                        insp_img_name = insp_img_name.split('#')[0]
                    else:
                        ref_img_name = ref_image_name.split('.')[0][-30:]
                        insp_img_name = insp_img_name.split('.')[0][-30:]

    #os.path.join(

                    filter_dir = f'./{args.filter_pairs_match_csv_path}/{args.valdataset}/{csv_name_path}/5'
                    os.makedirs(filter_dir, exist_ok=True)
                    fig_path = os.path.join(filter_dir, 
                                            f"label_{mclass_defect}_predict_{predict_mclass_defect}_{ref_img_name}_vs_{insp_img_name}.png")
                    
                    # fig_path = os.path.join('/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/240912_smt_defect_classification/jirra_filter', \
                    #                         '5',  f"label_{mclass_defect}_predict_{predict_mclass_defect}_{ref_image_name.split('#')[0]}_vs_{insp_img_name.split('#')[0]}.png")
                    
                    visualize_pair(ref_img, insp_img, predict_mclass_defect, mclass_defect, predict_multi_score, fig_path)
                    
                    filter_dir = f'./{args.filter_pairs_defect_path}/5/label_{mclass_defect}_predict_{predict_mclass_defect}'
                    os.makedirs(filter_dir, exist_ok=True)

                    fig_path = os.path.join(filter_dir,
                                            f"label_{mclass_defect}_predict_{predict_mclass_defect}_{ref_img_name}_vs_{insp_img_name}.jpg")
                    
                    if not os.path.exist(fig_path):
                        visualize_pair(ref_img, insp_img, predict_mclass_defect, mclass_defect, predict_multi_score, fig_path)

        print(f'{inference_model_mode} results: ')
        # compute top1 val accuracy and other metrics
        print(f'== {inference_model_mode} results: ==')
        acc_binary, acc_mclass, binary_multimetrics_np, mclass_multimetrics_np, precision_list, recall_list, f1score_list, thresholds_list, store_df = \
            evaluate_val_resultsnew(output_bos_np_softmax, output_bom_np_softmax, binary_labels, insp_labels,
                                    defect_decode, return_pr=True)
        store_df.columns = [f'{c}-th0.5' for c in store_df]
        f1score_list = [0 if np.isnan(a) else a for a in f1score_list]

        # prin higher threshold
        print(f'== threhsold = {higher_threshold} ==')
        indices = np.where(output_bos_np_softmax[:, 1] < higher_threshold) # 根据ng的分数是否高于阈值来决定
        output_bos_np_softmax_new = output_bos_np_softmax.copy()
        output_bom_np_softmax_new = output_bom_np_softmax.copy()

        ori_output_bos_np_softmax_new = output_bos_np_softmax_new.copy()
        ori_output_bom_np_softmax_new = output_bom_np_softmax_new.copy()

        output_bos_np_softmax_new[indices, 1] = 0
        output_bom_np_softmax_new[indices, 0] = 2

        binary_labels = binary_labels.astype(int)

        ori_binary_class = np.argmax(ori_output_bos_np_softmax_new, axis=1).astype(np.int8)
        ori_p_binary_class = np.max(ori_output_bos_np_softmax_new, axis=1)

        binary_class = np.argmax(output_bos_np_softmax_new, axis=1).astype(np.int8)
        p_binary_class = np.max(output_bos_np_softmax_new, axis=1)

        exact_defect_class = np.argmax(output_bom_np_softmax_new, axis=1).astype(np.int8)
        p_defect_class = np.max(output_bom_np_softmax_new, axis=1)

        ori_exact_defect_class = np.argmax(ori_output_bom_np_softmax_new, axis=1).astype(np.int8)
        ori_p_defect_class = np.max(ori_output_bom_np_softmax_new, axis=1)

        if args.show_high_thresh: # val_image_pair_data_raw  
            for index, img_names  in enumerate(zip(ref_image_name_list, insp_image_name_list)):
                binary_label = binary_labels[index]
                insp_label = insp_labels[index]
                csv_name_path = csv_names[index]
                csv_df = val_img_csv_dict[csv_name_path]

                if int(insp_label) < len(defect_decode) - 1:
                    mclass_defect = defect_decode[int(insp_label)]
                else:
                    raise

                predict_binary = binary_class[index]
                predict_binary_score = p_binary_class[index]
                predict_mclass = exact_defect_class[index]
                predict_multi_score = p_defect_class[index]

                ori_predict_binary = ori_binary_class[index]
                ori_predict_binary_score = ori_p_binary_class[index]

                ori_predict_mclass = ori_exact_defect_class[index]
                ori_predict_multi_score = ori_p_defect_class[index]

                if predict_mclass < len(defect_decode) - 1:
                    predict_mclass_defect = defect_decode[predict_mclass]
                    ori_predict_mclass_defect = defect_decode[ori_predict_mclass]
                else:
                    raise

                ref_image_sub_dir_path_ = ref_image_sub_dir_path[index]
                insp_image_sub_dir_path_ = insp_image_sub_dir_path[index]
                ref_image_name, insp_img_name = img_names
                
                ref_img_relative_path = os.path.join(ref_image_sub_dir_path_, ref_image_name)
                insp_img_relative_path = os.path.join(insp_image_sub_dir_path_, insp_img_name)

                csv_df_raws = csv_df[((csv_df['ref_image'] == ref_img_relative_path) & 
                                    (csv_df['insp_image'] == insp_img_relative_path))]
                # 当前这个pair对在csv中的位置
                ng_pair_index = val_image_pair_data.index[
                            (val_image_pair_data['ref_image'] == ref_img_relative_path) & 
                            (val_image_pair_data['insp_image'] == insp_img_relative_path)
                        ].values[0]
                
    

                # pair_name =  val_image_pair_data.loc[
                #         (val_image_pair_data['ref_image'] == ref_img_relative_path) & 
                #         (val_image_pair_data['insp_image'] == insp_img_relative_path), 
                #         'pairs_name'
                #     ].values[0]
                pair_name = str(ng_pair_index)+ '.jpg'
                val_image_pair_data.loc[
                        (val_image_pair_data['ref_image'] == ref_img_relative_path) & 
                        (val_image_pair_data['insp_image'] == insp_img_relative_path), 
                        'pairs_name'
                    ] = pair_name

                if predict_mclass_defect == mclass_defect:
                    csv_df_raws['model_predict_right'] = True
                    new_csv[csv_name_path].append(csv_df_raws)
                    val_image_pair_data.loc[
                        (val_image_pair_data['ref_image'] == ref_img_relative_path) & 
                        (val_image_pair_data['insp_image'] == insp_img_relative_path), 
                        'model_predict_right'
                    ] = True
                    if args.show_model_predict_wrong_or_right == 'wrong':
                        continue
                else:
                    csv_df_raws['model_predict_right'] = False
                    val_image_pair_data.loc[
                        (val_image_pair_data['ref_image'] == ref_img_relative_path) & 
                        (val_image_pair_data['insp_image'] == insp_img_relative_path), 
                        'model_predict_right'
                    ] = False
                    if args.show_model_predict_wrong_or_right == 'right':
                        continue

                new_csv[csv_name_path].append(csv_df_raws)
                
                ref_image_name, insp_img_name = img_names
                ref_img_path = os.path.join(image_folder, ref_image_sub_dir_path_, ref_image_name)
                insp_img_path = os.path.join(image_folder, insp_image_sub_dir_path_, insp_img_name)

                ref_img = cv2.imread(ref_img_path)
                insp_img = cv2.imread(insp_img_path)


                if args.lut_p == 1:
                    transform = transforms.Compose([ # 非同步的
                    LUT_VAL(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'), lut_p=lut_p),

                    ])  
                    ref_img = transform(ref_img)
                    insp_img = transform(insp_img)


                if '#' in ref_image_name:
                    ref_img_name = ref_image_name.split('#')[0]
                    insp_img_name = insp_img_name.split('#')[0]
                else:
                    ref_img_name = ref_image_name.split('.')[0][-30:]
                    insp_img_name = insp_img_name.split('.')[0][-30:]
                                                                                                            
                fig_path = os.path.join(predict_wrong_save_path, pair_name)
                insp_label = f'{predict_mclass_defect}_{predict_multi_score:.3f}'
                ori_model_predict_result = f'{ori_predict_mclass_defect}_{ori_predict_multi_score:.3f}_bi_{ori_predict_binary}_{ori_predict_binary_score:.3f}'
                visualize_pair(ref_img, insp_img, insp_label, mclass_defect, ori_model_predict_result, fig_path)
                
                if args.ues_gradcam:
                    gradCAMer = GradCAM(model, args.target_module, args.target_layer)
                    ref_image_cam = TransformImage(img_path=ref_img_path, rs_img_size_h=rs_img_size_h,
                                        rs_img_size_w=rs_img_size_w,
                                        ).transform()  # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p
                    insp_image_cam = TransformImage(img_path=insp_img_path, rs_img_size_h=rs_img_size_h,
                                            rs_img_size_w=rs_img_size_w,
                                            ).transform() # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p
                    # ref_image_cam, insp_image_cam = torch.FloatTensor(ref_image_cam).cuda(), torch.FloatTensor(insp_image_cam).cuda()

                    ref_image_cam = torch.ones_like(torch.FloatTensor(ref_image_cam)).cuda()
                    insp_image_cam = torch.ones_like(torch.FloatTensor(insp_image_cam)).cuda()

                    gradCAMer.forward(ref_image_cam, insp_image_cam, version_name, args.cam_type)

                    cam_contrastive_result_path = os.path.join(cam_result_save_dir, f'{ng_pair_index}_rel_{mclass_defect}_ori_pred_{ori_predict_mclass_defect}_{ori_predict_multi_score:.3f}_bi_{ori_predict_binary}_{ori_predict_binary_score:.3f}.png')
                    gradCAMer.show_contrastive_results(ref_img, insp_img, mclass_defect, insp_label, ori_model_predict_result, cam_contrastive_result_path)
                                                        # ref_img, insp_img, rel_insp_label, predict_insp_label, ori_model_predict_result, save_path

        val_image_pair_data.to_csv(os.path.join(predict_wrong_save_path, f'{args.valdataset}_{args.version_folder}_{args.split_get_ng_or_ok}.csv'))

        acc_binary_new, acc_mclass_new, binary_multimetrics_np_new, mclass_multimetrics_np_new, store_df_th = \
            evaluate_val_resultsnew(output_bos_np_softmax_new, output_bom_np_softmax_new,
                                    binary_labels, insp_labels,
                                    defect_decode, return_pr=False)
        store_df_th.columns = [f'{c}-th{higher_threshold}' for c in store_df_th]
        store_df_all = pd.concat([store_df, store_df_th], axis=1)
        store_df_all['val_dataset'] = args.valdataset
        store_df_all['version_folder'] = version_folder
        store_df_all['model_name'] = version_name
        store_df_all['region'] = region
        store_df_all['backbone'] = backbone_arch
        store_df_all['ckp_epoch'] = ckp_epoch

        all_store_df_list.append(store_df_all)
        # save pr curve and f1 thre curve
        fig, axes = plt.subplots(2, 1, figsize=(4, 6))
        axes[0].plot(recall_list, precision_list, 'r-')
        # axes[0].set_title(f'PR curve')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precesion')

        axes[1].plot(thresholds_list, f1score_list, 'b-')
        # axes[1].set_title(f'F1-Thre curve')
        max_f1 = np.max(f1score_list)
        max_f1_thre = thresholds_list[np.argmax(f1score_list)]
        axes[1].plot(max_f1_thre, max_f1, 'ro', label=f'f1={max_f1:.3f} \nth={max_f1_thre:.3f}')
        axes[1].legend(loc=2)
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('F1 score')
        plt.tight_layout()
        plt.savefig(os.path.join(compare_version_res_folder, version_name.split('.pth.tar')[0] + '.jpg'))
        plt.close()

        # save metric results
        compare_versions['bi_acc'].append(acc_binary)
        compare_versions['mclass_acc'].append(acc_mclass)
        # binary_label =0: ok, =1: defect
        binary_class_list = ['ok', 'ng']
        for i, bi_c in enumerate(binary_class_list):
            multimetric_value = binary_multimetrics_np[i]
            for key, value in multimetric_value.items():
                compare_versions[f'bi_{bi_c}_{key}'].append(value)

        for i, mi_c in defect_decode.items():
            multimetric_value = mclass_multimetrics_np[i]
            for key, value in multimetric_value.items():
                if mi_c in ['ok', 'ng']:
                    compare_versions[f'mclass_{mi_c}_{key}'].append(value)

        outputs_combined = np.vstack([binary_class, p_binary_class, exact_defect_class, p_defect_class]).T
        outputs_combined_df = pd.DataFrame(outputs_combined,
                                           columns=['binary_class', 'p_binary_class', 'defect_class',
                                                    'p_defect_class'])
        binary_label_df = pd.DataFrame(binary_labels, columns=['binary_y'])

        # compute top1 val accuracy and other metrics for unique insp samples
        # prepare unique sample
        insp_image_name_array = np.array(insp_image_name_list)
        insp_image_name_unique, insp_image_name_counts = np.unique(insp_image_name_list, return_counts=True)
        output_bos_class_unique_mode = []
        output_bos_class_unique_strict = []
        output_bom_class_unique = []
        binary_labels_unique = []
        insp_labels_unique = []
        output_bos_class = np.argmax(output_bos_np_softmax, axis=1)[:, None]
        output_bom_class = np.argmax(output_bom_np_softmax, axis=1)[:, None]

        for insp_image_name in insp_image_name_unique:
            indices = np.where(insp_image_name_array == insp_image_name)
            output_bos_i = output_bos_class[indices]
            output_bom_i = output_bom_class[indices]
            binary_labels_i = binary_labels[indices]
            insp_labels_i = insp_labels[indices]

            if len(insp_labels_i) > 1:
                output_bos_i_mode = np.median(output_bos_i)
                if len(np.unique(output_bos_i)) > 1:
                    bos_label, counts = np.unique(output_bos_i, return_counts=True)
                    output_bos_i_strict = bos_label[np.argmin(counts)]
                else:
                    output_bos_i_strict = output_bos_i_mode
                output_bos_class_unique_mode.append(output_bos_i_mode)
                output_bos_class_unique_strict.append(output_bos_i_strict)
                output_bom_class_unique.append(np.median(output_bom_i))
                binary_labels_unique.append(np.unique(binary_labels_i)[0])
                insp_labels_unique.append(np.unique(insp_labels_i)[0])
            else:
                output_bos_class_unique_mode.append(output_bos_i[0, 0])
                output_bos_class_unique_strict.append(output_bos_i[0, 0])
                output_bom_class_unique.append(output_bom_i[0, 0])
                binary_labels_unique.append(binary_labels_i[0, 0])
                insp_labels_unique.append(insp_labels_i[0, 0])

        output_bos_class_unique_array = np.array(output_bos_class_unique_mode)[:, None]
        output_bom_class_unique_array = np.array(output_bom_class_unique)[:, None]
        binary_labels_unique_array = np.array(binary_labels_unique)[:, None]
        insp_labels_unique_array = np.array(insp_labels_unique)[:, None]
        if print_unique:
            print(f'== unique sample results mode: ==')
            acc_binary_unique, acc_mclass_unique, binary_multimetrics_np_unique, mclass_multimetrics_np_unique = \
                evaluate_val_resultsnew(output_bos_class_unique_array, output_bom_class_unique_array,
                                        binary_labels_unique_array, insp_labels_unique_array,
                                        defect_decode, class_input=True, return_pr=False)

        trt_model_outputs['binary_outputs'][region] = model_outputs['binary_outputs'][region]
        # trt_model_outputs['mclass_outputs'][region] = output_bom_np_softmax
        trt_model_outputs['binary_labels'][region] = model_outputs['binary_labels'][region]
        trt_model_outputs['mclass_labels'][region] = model_outputs['mclass_labels'][region]
        trt_model_outputs['insp_image_names'][region] = model_outputs['insp_image_names'][region]

        print('hold')
        val_image_pair_res_df = pd.DataFrame(val_image_pair_res)
        val_image_pair_res_df.to_csv(
            os.path.join(compare_version_res_folder, version_name.split('.pth.tar')[0] + f'{rs_img_size_w}{rs_img_size_h}.csv'))
    #         with open(os.path.join(os.path.join(compare_version_res_folder, version_name.split('.pth.tar')[0]), 'wb') as handle:
    #             pickle.dump(val_image_pair_res, handle)

    # compute top1 val accuracy and other metrics

    # compare_versions_df.to_csv(compare_version_csv_name)

    Visualize = args.Visualize
    if Visualize:
        import matplotlib.pyplot as plt

        idx = 0
        binary_label_dict = ['ok', 'ng']
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for ref_image_batch, insp_image_batch in zip(ref_image_batches, insp_image_batches):
            for img_ref, img_insp in zip(ref_image_batch, insp_image_batch):
                img_ref = np.transpose(img_ref, (1, 2, 0))
                img_ref = np.clip((img_ref * std + mean) * 255, 0, 255).astype(np.uint8)
                img_insp = np.transpose(img_insp, (1, 2, 0))
                img_insp = np.clip((img_insp * std + mean) * 255, 0, 255).astype(np.uint8)
                true_binary = binary_labels.flatten()[idx]
                pred_binary = binary_class[idx]
                p_ng = output_bos_np_softmax[idx, 1]
                true_label = insp_labels.flatten()[idx]
                pred_label = exact_defect_class[idx]
                p_mclass = output_bom_np_softmax[idx, pred_label]
                ref_image_name = ref_image_name_list[idx][:-4]
                insp_image_name = insp_image_name_list[idx][:-4]
                idx += 1
                # if (true_label > 0) & (true_label == 2):
                pred_binary_tmpt = p_ng > higher_threshold
#                 if (true_binary != pred_binary_tmpt):
                if (true_binary):

                    # plot reference image against inspected image
                    fig, axes = plt.subplots(2, 1, figsize=(4, 6))

                    axes[0].imshow(img_ref, cmap='gray')
                    axes[0].set_title(f'img {idx} ok sample/binary res '
                                      f'\n true ={binary_label_dict[true_binary]}'
                                      f'\n predicted ={binary_label_dict[pred_binary]} (p={p_ng:.3f}) ')
                    axes[0].set_xticks([])
                    axes[0].set_yticks([])

                    axes[1].imshow(img_insp, cmap='gray')
                    axes[1].set_title(f'img {id} ng sample/mclass res  '
                                      f'\n true ={defect_decode[true_label]}, '
                                      f'\n predicted ={defect_decode[pred_label]} (p={p_mclass:.3f}) ')
                    axes[1].set_xticks([])
                    axes[1].set_yticks([])

                    os.makedirs(os.path.join(compare_version_res_folder, f'{image_saving_phase}_{region}'),
                                exist_ok=True)
                    plt.savefig(os.path.join(compare_version_res_folder, f'{image_saving_phase}_{region}',
                                             f'{insp_image_name}_vs_{ref_image_name}_{true_label}.jpg'))

                    plt.close()

    all_store_df_res = pd.concat(all_store_df_list).reset_index(drop=True)
    all_store_df_res.to_csv(os.path.join(ckp_folder, f'{region}_{args.valdataset}_{version_folder}_{rs_img_size_w}{rs_img_size_h}_lutp{lut_p}.csv'))#f'./csv/{region}_{args.valdataset}_{version_folder}_{rs_img_size_w}{rs_img_size_h}_lutp{lut_p}.csv'))