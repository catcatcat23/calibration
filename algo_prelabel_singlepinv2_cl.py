import glob
import pickle
import pandas as pd
import torch
import time
import os
import random


import numpy as np
from utils.utilities import TransformImage, split
from utils.metrics import multiclass_multimetric, accuracy, evaluate_val_resultsnew
from scipy.special import softmax
import pickle
import gc
import matplotlib.pyplot as plt
from util import filter_from_df
from algo_test_utils import get_test_all_csvs, get_test_df
import warnings
from torchvision import transforms
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser(description='Run inference to check res') #  padgroupv1.22.6lut1select padgroupv1.22.5select = 0.81 singlepadlut0 = 0.77
parser.add_argument('--version_folder', default='singlepinpadv2.17.3select', type=str) #singlepadv0.6impl   singlepadv0.9 padgroupv1.10rgbwhite
parser.add_argument('--version', default='v2.17.3', type=str)  #v0.12final
parser.add_argument('--confidence', default='certain', type=str)
parser.add_argument('--output_type', default='dual2', type=str)
parser.add_argument('--valdataset', default='fzqzzl', type=str, help='bbtest, bbtestmz, alltestval, newval, debug, cur_jiraissue, led_cross_pair, jiraissues')
parser.add_argument('--date', default='241022', type=str)   # 241022   debug-240723  body-241023_white  bbtest，bbtestmz没有图
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--threshold', default=0.8, type=float, help='higher_threshold to check')
parser.add_argument('--val_img_file_path', default=f'annotation_test_labels_pairs_region_output_type_flexible.csv',type=str)
parser.add_argument('--lut_p', default=0, type=float)

parser.add_argument('--region', default=f'singlepinpad', type=str)   # padgroup
parser.add_argument('--Visualize', default=0, type=int)
parser.add_argument('--light_device', default='all', type=str, choices=['2d', '3d', 'all'])
parser.add_argument('--img_color', default='rgb', type=str, choices=['rgb', 'white'])
parser.add_argument('--dropdup', default=True, type=bool)
parser.add_argument('--img_type', default='png', type=str)

parser.add_argument('--aug_img', default=False, type=bool)
parser.add_argument('--colorjitter', default=[0.1, 0.1, 0.15, 0.1], type=list)
parser.add_argument('--sharpness_factor', default=4, type=int)
parser.add_argument('--sharpness_p', default=1, type=float)
parser.add_argument('--calibration_T', default=1.0, type=float)

parser.add_argument('--inference_model_mode', default='torch', type=str, choices=['torch', 'onnx_slim'])
parser.add_argument('--save_metric_results', default=True, type=bool)
parser.add_argument('--inference_high_thres', default=True, type=bool)

args = parser.parse_args()

print(f"version: {args.version}")
img_color = args.img_color
print(f"img_color: {img_color}")

compare_trt = False

inference_model_mode = args.inference_model_mode
print(f"inference_model_mode: {inference_model_mode}")
seed = 42

# ckp_folder = '/mnt/pvc-nfs-dynamic/robinru/model_ckps/aoi-sdkv0.10.0'
ckp_folder = f'./models/checkpoints/{args.region}/{args.version}/{args.version_folder}'
# ckp_folder = '/mnt/pvc-nfs-dynamic/robinru/model_ckps/aoi-sdkv0.10.0/padgroupv1.8.0'

region_list = args.region.split(';')
if 'component' in region_list or 'body' in region_list:
    image_folder =  '/mnt/dataset/xianjianming/data_clean_white/' #'/mnt/ssd/classification/data/data_clean_white/'
else:
    image_folder = '/mnt/dataset/xianjianming/data_clean/' #'/mnt/ssd/classification/data/data_clean/'

if args.valdataset == 'debug':
    image_folder = '/mnt/dataset/xianjianming/debug/' #'/mnt/ssd/classification/data/data_clean/'

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

compare_version_res_folder = f'./results/{args.region}/{version_folder}_{date}/{args.valdataset}_T{args.calibration_T}'
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
        val_image_pair_path_list = get_test_all_csvs(image_folder, date,  'padgroup', args.valdataset, img_color)

    elif region == 'single_pin' or region == 'singlepinpad':
        anno_folder = os.path.join(image_folder, 'merged_annotation', date)
        if 'rs' in version_folder:
            wh = version_folder.split('rs')[-1]
            wh_length = len(wh)
            rs_img_size_w, rs_img_size_h = int(wh[:wh_length//2]), int(wh[wh_length//2:])
        else:
            rs_img_size_w = 128
            rs_img_size_h = 32
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'
        # algo_csvs = get_test_all_csvs(image_folder, date,  'singlepinpad')
        val_image_pair_path_list = get_test_all_csvs(image_folder, date,  'singlepinpad', args.valdataset, img_color)
                        
        
    elif region == 'singlepad':
        anno_folder = os.path.join(image_folder, 'merged_annotation', date)
        if 'rs' in version_folder:
            wh = version_folder.split('rs')[-1]
            wh_length = len(wh)
            rs_img_size_w, rs_img_size_h = int(wh[:wh_length//2]), int(wh[wh_length//2:])
        else:  
            rs_img_size_w = 64
            rs_img_size_h = 64
        
        print(f"rs_img_size_w = {rs_img_size_w} \n rs_img_size_h = {rs_img_size_h}")
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'
        # algo_csvs = get_test_all_csvs(image_folder, date,  'singlepad')
        val_image_pair_path_list = get_test_all_csvs(image_folder, date,  'singlepad', args.valdataset, img_color)
        
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

        # algo_csvs = get_test_all_csvs(image_folder, date,  'body')
        val_image_pair_path_list = get_test_all_csvs(image_folder, date,  'body', args.valdataset, img_color)
    print(f'rsstr: {rsstr}')
    val_image_pair_path_list = list(set(val_image_pair_path_list))
    print(args.valdataset, val_image_pair_path_list)

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

    # 数据增强方式
    from dataloader.image_resampler_pins import LUT_VAL
    img_folder = '/mnt/pvc-nfs-dynamic/xianjianming/data/'
    if args.aug_img:
        sharpness_p = args.sharpness_p
        sharpness_save =args.region + '-' + str(args.sharpness_factor) + '-' + str(args.sharpness_p)
        # os.makedirs(sharpness_save, exist_ok=True)
        # os.makedirs(sharpness_save + '-ori', exist_ok=True)
        transform = transforms.Compose([
                                    LUT_VAL(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'), lut_p=lut_p),

                                        # transforms.ColorJitter(
                                        #     brightness=args.colorjitter[0], contrast=args.colorjitter[1],
                                        #     saturation=args.colorjitter[2], hue=args.colorjitter[3]
                                        # ),
                                        transforms.RandomAdjustSharpness(sharpness_factor=args.sharpness_factor, p=1)
                                     ])
            
    else:
        transform = transforms.Compose([ # 非同步的
            LUT_VAL(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'), lut_p=lut_p),

        ])  
        sharpness_save = None
        sharpness_p = 0

        
    # decode the defect id back to defect label
    n_class, insp_label_list, ref_image_name_list, defect_code, defect_decode, defect_full_decode, val_image_pair_res, binary_labels, ref_labels, \
           insp_labels, ref_image_batches, insp_image_batches, insp_image_name_list = get_test_df(args, 
                                                                            val_image_pair_path_list, 
                                                                            region, 
                                                                            version_folder, 
                                                                            batch_size, 
                                                                            image_folder,
                                                                            transform,
                                                                            sharpness_save,
                                                                            sharpness_p = sharpness_p,
                                                                            rs_img_size_w = rs_img_size_w,
                                                                            rs_img_size_h = rs_img_size_h,)

    for version_name in version_name_list:
        backbone_arch = version_name.split('rs')[0].split(region)[-1]
        if 'nb' not in version_name:
            n_units = [128, 128]
        else:
            if 'top' in version_name and 'Cosin' in version_name:
                n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                           version_name.split('f16')[-1].split('top')[0].split('nb')[-1].split('Cosin')[0].split('nm')]
            elif 'top' in version_name and 'CL' in version_name:
                n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                           version_name.split('f16')[-1].split('top')[0].split('nb')[-1].split('CL')[0].split('nm')]
            elif 'top' in version_name:
                n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                           version_name.split('f16')[-1].split('top')[0].split('nb')[-1].split('nm')]
            elif 'Cosin' in version_name:
                n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                           version_name.split('f16')[-1].split('last')[0].split('nb')[-1].split('Cosin')[0].split('nm')]
            elif 'CL' in version_name:
                n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                           version_name.split('f16')[-1].split('last')[0].split('nb')[-1].split('CL')[0].split('nm')]
            else:
                n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                           version_name.split('f16')[-1].split('last')[0].split('nb')[-1].split('nm')]
        
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
                output_type = "CLLdual2"
            elif 'CL' in version_name:
                output_type = "CLdual2"
            else:
                output_type = "dual2"

            # define model and load check point
            if 'cbam' in backbone_arch and 'CL' not in version_name:
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
            elif 'cbam' in backbone_arch and 'CL' in version_name:
                print(backbone_arch)
                if region == 'body' or region == 'component':
                    attention_list_str = backbone_arch.split('cbam')[-1]
                    if len(attention_list_str.split('cbam')[-1]) == 0:
                        attention_list = None
                    else:
                        attention_list = [int(attention_list_str)]
                
                from models.MPB3_attention_CL import MPB3net
                print(attention_list)

                model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                                  output_form=output_type, attention_list=attention_list)           
            
            elif 'CL' in version_name:
                from models.MPB3_CL import MPB3net
                model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units, output_form=output_type)

            elif args.region == 'body' and 'merge' in version_name:

                from models.MPB3_BODY import MPB3net

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

                output_bos = output_bos / args.calibration_T
                output_bom = output_bom / args.calibration_T

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

                time_end = time.time()
                batch_run_time = time_end - time_start
                print(f'avg: {batch_run_time * 1000}ms, batch_size: {batch_size}')

                output_bos_np_list.append(output_bos_np_batch / args.calibration_T)
                output_bom_np_list.append(output_bom_np_batch / args.calibration_T)
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
        print(f'{inference_model_mode} results: ')
        # print(f'bos: {output_bos_np_softmax}')
        # print(f'bom: {output_bom_np_softmax}')

        # compute top1 val accuracy and other metrics
        print(f'== {inference_model_mode} results: ==')
        acc_binary, acc_mclass, binary_multimetrics_np, mclass_multimetrics_np, precision_list, recall_list, f1score_list, thresholds_list, store_df = \
            evaluate_val_resultsnew(output_bos_np_softmax, output_bom_np_softmax, binary_labels, insp_labels,
                                    defect_decode, return_pr=True)
        store_df.columns = [f'{c}-th0.5' for c in store_df]
        f1score_list = [0 if np.isnan(a) else a for a in f1score_list]
            
        if args.inference_high_thres:
            # prin higher threshold
            print(f'== threhsold = {higher_threshold} ==')
            indices = np.where(output_bos_np_softmax[:, 1] < higher_threshold)
            output_bos_np_softmax_new = output_bos_np_softmax.copy()
            output_bom_np_softmax_new = output_bom_np_softmax.copy()
            output_bos_np_softmax_new[indices, 1] = 0
            output_bom_np_softmax_new[indices, 0] = 2

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
        plt.savefig(os.path.join(compare_version_res_folder, version_name.split('.pth.tar')[0] + f'.jpg'))
        plt.close()

        if args.save_metric_results:
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

    if args.save_metric_results:
        all_store_df_res = pd.concat(all_store_df_list).reset_index(drop=True)
        save_path = os.path.join(ckp_folder, f'{region}_{args.valdataset}_{version_folder}_{rs_img_size_w}{rs_img_size_h}_lutp{lut_p}_{args.light_device}_{img_color}_{args.img_type}_augimg{args.aug_img}_sharp{args.sharpness_factor}_calibration_T{args.calibration_T}_threshold{args.threshold}.csv')
        print(f'save to: {save_path}')
        all_store_df_res.to_csv(save_path)#f'./csv/{region}_{args.valdataset}_{version_folder}_{rs_img_size_w}{rs_img_size_h}_lutp{lut_p}.csv'))