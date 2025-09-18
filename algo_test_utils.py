import os
import time
import torch
import random

import numpy as np
import pandas as pd

from util import filter_from_df
from utils.utilities import TransformImage, split, TransformImageFusion
from utils.metrics import multiclass_multimetric, accuracy, p_r_curve

from scipy.special import softmax

def get_ece_black_test_df(args, val_image_pair_path_list, region, 
                version_folder, batch_size, test_image_folder, transform=None, 
                sharpness_save = None, sharpness_p = 0, calibrate = False):
    
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
        rs_img_size_w = 224
        rs_img_size_h = 224
        rsstr = f'rs{rs_img_size_w}'

        if 'slim' in version_folder:
#             defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
            defect_code = {'ok': 0, 'wrong': 3}  # v1.32
            defect_code_slim = defect_code
            defect_sdk_decode = {'ok': 2, 'wrong': 4}
        else:
            defect_code = {'ok': 0, 'missing': 1, 'wrong': 3}  # v1.32
            defect_code_slim = defect_code
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4}
        

    elif region == 'single_pin' or region == 'singlepinpad':
        if 'rs' in version_folder:
            wh = version_folder.split('rs')[-1]
            wh_length = len(wh)
            rs_img_size_w, rs_img_size_h = int(wh[:wh_length//2]), int(wh[wh_length//2:])
        else:
            rs_img_size_w = 128
            rs_img_size_h = 32
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'

        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
            
        defect_code_slim = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_sdk_decode = {'ok': 2, 'undersolder': 3, 'pseudosolder': 4}

    elif region == 'singlepad':
        if 'rs' in version_folder:
            wh = version_folder.split('rs')[-1]
            wh_length = len(wh)
            rs_img_size_w, rs_img_size_h = int(wh[:wh_length//2]), int(wh[wh_length//2:])
        else:  
            rs_img_size_w = 64
            rs_img_size_h = 64
        
        print(f"rs_img_size_w = {rs_img_size_w} \n rs_img_size_h = {rs_img_size_h}")
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'

        defect_code_slim = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_sdk_decode = {'ok': 2, 'undersolder': 3, 'pseudosolder': 4}

    elif region == 'pins_part' or region == 'padgroup':
        if '_' in version_folder:
            rs_img_size_w = int(version_folder.split('_')[1])
            rs_img_size_h = int(version_folder.split('_')[2])
#             img_color = version_folder.split('_')[3]
            # img_color = 'rgb'
        else:
            rs_img_size_w, rs_img_size_h = 128, 128
            # img_color = 'rgb'
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'

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

    defect_full_decode = {v: k for k, v in defect_sdk_decode.items()}
    n_class = len(defect_code)
    # defect_decode = {i: v for i, v in enumerate(defect_list)}
    defect_decode = {i: k for i, k in enumerate(defect_code.keys())}
    defect_convert = {v: i for i, v in enumerate(defect_code.values())}
    defect_decode.update({-1: 'ng'})
    # load test images
    val_image_pair_data_list = [pd.read_csv(path) for path in val_image_pair_path_list]
    val_image_pair_data_raw = pd.concat(val_image_pair_data_list).reset_index(drop=True)

    val_image_pair_data_raw = val_image_pair_data_raw.drop_duplicates(subset=['ref_image', 'insp_image']).reset_index(drop=True)

    # val_image_pair_data_raw_has_img_name = val_image_pair_data_raw.dropna(subset=['ref_image_name'])
    # val_image_pair_data_raw_no_img_name = val_image_pair_data_raw[val_image_pair_data_raw['ref_image_name'].isna()]

    # val_image_pair_data_raw_has_img_name = val_image_pair_data_raw_has_img_name.drop_duplicates(subset=['ref_image_name', 'insp_image_name']).reset_index(drop=True)
    # val_image_pair_data_raw_no_img_name = val_image_pair_data_raw_no_img_name.drop_duplicates(subset=['ref_image', 'insp_image']).reset_index(drop=True)

    if region == 'body' and 'slim' in version_folder:
        val_image_pair_data_raw['insp_y_raw_old'] = list(val_image_pair_data_raw['insp_y_raw'])
        val_image_pair_data_raw['insp_y_raw'] = [3 if y == 1 else y for y in val_image_pair_data_raw['insp_y_raw_old']]  

    val_image_pair_data = val_image_pair_data_raw[
            [y in list(defect_code_slim.values()) for y in val_image_pair_data_raw['insp_y_raw']]]
    val_image_pair_data['insp_y'] = [defect_convert[yraw] for yraw in val_image_pair_data['insp_y_raw']]
    
    if args.confidence == 'certain':
        val_image_pair_data = val_image_pair_data[
            (val_image_pair_data['confidence'] == 'certain') | (
                    val_image_pair_data['confidence'] == 'unchecked')].copy().reset_index(drop=True)
    elif args.confidence == 'uncertain':
        val_image_pair_data = val_image_pair_data[
            (val_image_pair_data['confidence'] == 'uncertain')].copy().reset_index(drop=True)
    elif args.confidence == 'all':
        val_image_pair_data = val_image_pair_data[
            (val_image_pair_data['confidence'] != 'BAD_PAIR')].copy().reset_index(drop=True)
    
    val_image_pair_data = filter_from_df(val_image_pair_data, args.light_device)
    print(len(val_image_pair_data))
#     val_image_pair_data = val_image_pair_data[:600]
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
    ref_image_label_batches = []
    insp_image_label_batches = []
    binary_label_batchs = []

    n_batches = int(len(val_image_pair_data) / batch_size)
    if len(val_image_pair_data) < batch_size:
        batch_idices_list = [range(len(val_image_pair_data))]
        n_batches = 1
    else:
        batch_idices_list = split(list(val_image_pair_data.index), n_batches)
        
    for batch_indices in batch_idices_list:
        
        ref_batch = []
        insp_batch = []
        if calibrate:
            ref_batch_label = []
            insp_batch_label = []
            per_binary_label = []

        for i in batch_indices:

            if counter % 1000 == 0:
                print(f'{counter}')
            counter += 1
            val_image_pair_i = val_image_pair_data.iloc[i]

            ref_image_path = os.path.join(test_image_folder, val_image_pair_i['ref_image'])
            insp_image_path = os.path.join(test_image_folder, val_image_pair_i['insp_image'])

            ref_image_name = val_image_pair_i['ref_image'] #.split('/')[-1]
            insp_image_name = val_image_pair_i['insp_image'] # .split('/')[-1]
            # scale, normalize and resize test images
            if random.random() < sharpness_p:
                transform_in = transform
            else:
                transform_in = None

            ref_image = TransformImage(img_path=ref_image_path, img_type = args.img_type,  rs_img_size_h= rs_img_size_h,
                                       rs_img_size_w = rs_img_size_w, transform=transform_in, sharpness_save =sharpness_save
                                      ).transform()  # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p
            insp_image = TransformImage(img_path=insp_image_path, img_type = args.img_type, rs_img_size_h=rs_img_size_h,
                                        rs_img_size_w=rs_img_size_w, transform=transform_in, sharpness_save =sharpness_save
                                        ).transform() # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p

            # get ground truth labels
            ref_label = val_image_pair_i['ref_y']
            insp_label = val_image_pair_i['insp_y']
            ref_batch.append(ref_image)
            insp_batch.append(insp_image)

            ref_label_list.append(ref_label)
            insp_label_list.append(insp_label)

            if calibrate:
                per_binary_label.append(val_image_pair_i['binary_y'])
                
                ref_batch_label.append(ref_label)
                insp_batch_label.append(insp_label)

            ref_image_name_list.append(ref_image_name)
            insp_image_name_list.append(insp_image_name)

        ref_image_batches.append(np.concatenate(ref_batch, axis=0))
        insp_image_batches.append(np.concatenate(insp_batch, axis=0))

        if calibrate:
            ref_image_label_batches.append(np.array(ref_batch_label))
            insp_image_label_batches.append(np.array(insp_batch_label))
            binary_label_batchs.append(np.array(per_binary_label))

    # n_batches = np.max([int(len(ref_image_list) / batch_size), 1])

    binary_labels = np.array(binary_label_list).reshape([-1, 1])
    ref_labels = np.array(ref_label_list).reshape([-1, 1])
    insp_labels = np.array(insp_label_list).reshape([-1, 1])

    if calibrate:
        return val_image_pair_data, n_class, insp_label_list, ref_image_name_list, defect_code, defect_decode, defect_full_decode, val_image_pair_res, binary_labels, ref_labels, \
           insp_labels, ref_image_batches, insp_image_batches, insp_image_name_list, ref_image_label_batches, insp_image_label_batches, binary_label_batchs
    else:
        return val_image_pair_data, n_class, insp_label_list, ref_image_name_list, defect_code, defect_decode, defect_full_decode, val_image_pair_res, binary_labels, ref_labels, \
           insp_labels, ref_image_batches, insp_image_batches, insp_image_name_list

def evaluate_val_resultsnew(output_bos_np_softmax, output_bom_np_softmax, binary_labels, insp_labels, defect_decode,
                         class_input=False, return_pr=False):
    # res_dict = {}
    # compute accuracy
    acc_binary = accuracy(output_bos_np_softmax, binary_labels, input_type='np', class_input=class_input)
    acc_mclass = accuracy(output_bom_np_softmax, insp_labels, input_type='np', class_input=class_input)
    # res_string_to_print = f'binary accuracy = {acc_binary:.5f}\n'
    # res_string_to_print += f'mclass accuracy = {acc_mclass:.5f}\n'
    # res_dict['binary_accuracy'] = acc_binary
    # res_dict['mclass_accuracy'] = acc_binary

    # # compute other metrics
    # # check other metrics:
    # # binary_label =0: ok, =1: defect
    # binary_class_list = ['ok', 'ng']
    # binary_class_number = list(range(len(binary_class_list)))
    # res_string_to_print += '== Binary output ==\n'
    # binary_multimetrics_np = multiclass_multimetric(output_bos_np_softmax, binary_labels.flatten(), binary_class_number,
    #                                                 'np', class_input=class_input)
    # n_binary_dic = {bi_c: np.sum(binary_labels.flatten() == i) for i, bi_c in enumerate(binary_class_list)}

    # res_string_to_print += f'weighted f1 score = ' + str(binary_multimetrics_np['weight_f1_score']) + '\n'
    # res_dict['binary_weighted_f1_score'] = binary_multimetrics_np['weight_f1_score']
    # for i, bi_c in enumerate(binary_class_list):
    #     multimetric_value = binary_multimetrics_np[i]
    #     multimetric_value_str_list = [f'{key}:{value:.3f}' for key, value in multimetric_value.items()]
    #     res_string_to_print += f'{bi_c} (n={n_binary_dic[bi_c]}): ' + ', '.join(multimetric_value_str_list) + '\n'
    #     for key, value in multimetric_value.items():
    #         res_dict[f'bi_{bi_c}_{key}'] = value

    # mclass_class_number = list(range(len(defect_decode.keys())))
    # mclass_multimetrics_np = multiclass_multimetric(output_bom_np_softmax, insp_labels.flatten(), mclass_class_number,
    #                                                 'np', class_input=class_input)

    # res_string_to_print += '== Multiclass output == \n'
    # res_string_to_print += f'weighted f1 score = ' + str(mclass_multimetrics_np['weight_f1_score']) + '\n'
    # res_dict['mclass_weighted_f1_score'] = mclass_multimetrics_np['weight_f1_score']

    # n_mclass_dic = {}
    # for i, mi_c in defect_decode.items():
    #     multimetric_value = mclass_multimetrics_np[i]
    #     multimetric_value_str_list = [f'{key}:{value:.3f}' for key, value in multimetric_value.items()]
    #     n_mclass_dic[mi_c] = np.sum(insp_labels.flatten() == i)
    #     res_string_to_print += f'{mi_c} (n={n_mclass_dic[mi_c]}): ' + ', '.join(multimetric_value_str_list) + '\n'
    #     for key, value in multimetric_value.items():
    #         res_dict[f'mc_{mi_c}_{key}'] = value

    # # print(res_string_to_print)
    # res_dict['res_string'] = res_string_to_print
    # res_df = pd.DataFrame(res_dict, index=[0])
    # compute pr curve
    # if not class_input and return_pr:
    #     y_scores = output_bos_np_softmax[:, 1]
    #     y_true = binary_labels.flatten()
    #     precision_list, recall_list, f1score_list, thresholds_list = p_r_curve(y_true, y_scores)

    # if return_pr:
    #     return acc_binary, acc_mclass, binary_multimetrics_np, mclass_multimetrics_np, precision_list, recall_list, f1score_list, thresholds_list, res_df
    # else:
    return acc_binary, acc_mclass #, binary_multimetrics_np, mclass_multimetrics_np, res_df
    
def get_test_df_results(model, ref_image_batches, insp_image_batches, insp_labels, binary_labels,
                        version_name, higher_threshold):
    model.eval()
    # compute ouputs
    output_bos_list = []
    output_bom_list = []
    # print(f"get_test_df_results || len(ref_image_batches) = {len(ref_image_batches)}")
    for img1, img2 in zip(ref_image_batches, insp_image_batches):

        img1, img2 = torch.FloatTensor(img1).cuda(), torch.FloatTensor(img2).cuda()
        # print(f"img1.shape = {img1.shape}")
        # print(f"img2.shape = {img1.shape}")

        with torch.no_grad():
            if 'CL' in version_name:
                output_bos, output_bom, _, _ = model(img1, img2)
            else:
                output_bos, output_bom = model(img1, img2)

        # update record
        # print(f"output_bos.shape = {output_bos.shape}")
        output_bos_list.append(output_bos)
        output_bom_list.append(output_bom)

    # print(f"get_test_df_results || len(output_bos_list) = {len(output_bos_list)}")
    # print(f"get_test_df_results || len(output_bom_list) = {len(output_bom_list)}")

    output_bos_all = torch.cat(output_bos_list, dim=0)
    output_bom_all = torch.cat(output_bom_list, dim=0)
    output_bos_np = output_bos_all.detach().cpu().numpy()
    output_bom_np = output_bom_all.detach().cpu().numpy()
    # apply softmax
    output_bos_np_softmax = softmax(output_bos_np, axis=1)
    output_bom_np_softmax = softmax(output_bom_np, axis=1)

    binary_labels = binary_labels.astype(int)

    indices = np.where(output_bos_np_softmax[:, 1] < higher_threshold)
    output_bos_np_softmax_new = output_bos_np_softmax.copy()
    output_bom_np_softmax_new = output_bom_np_softmax.copy()
    output_bos_np_softmax_new[indices, 1] = 0
    output_bom_np_softmax_new[indices, 0] = 2

    # print(f"output_bos_np_softmax_new shape = {output_bos_np_softmax_new.shape}")
    # print(f"output_bom_np_softmax_new shape = {output_bom_np_softmax_new.shape}")

    acc_binary_new = accuracy(output_bos_np_softmax_new, binary_labels, input_type='np', class_input=False)
    acc_mclass_new = accuracy(output_bom_np_softmax_new, insp_labels, input_type='np', class_input=False)
    
    return acc_binary_new, acc_mclass_new

def get_test_df(args, val_image_pair_path_list, region, 
                version_folder, batch_size, test_image_folder, transform=None, 
                sharpness_save = None, sharpness_p = 0, calibrate = False, rs_img_size_w = None, rs_img_size_h = None, fusion = None):
    print(f'sharpness_p = {sharpness_p}')
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
        if rs_img_size_w is None:
            rs_img_size_w = 224
        if rs_img_size_h is None:
            rs_img_size_h = 224
        rsstr = f'rs{rs_img_size_w}'

        if 'slim' in version_folder:
#             defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
            defect_code = {'ok': 0, 'wrong': 3}  # v1.32
            defect_code_slim = defect_code
            defect_sdk_decode = {'ok': 2, 'wrong': 4}
        else:
            defect_code = {'ok': 0, 'missing': 1, 'wrong': 3}  # v1.32
            defect_code_slim = defect_code
            defect_sdk_decode = {'ok': 2, 'missing': 3, 'wrong': 4}
        

    elif region == 'single_pin' or region == 'singlepinpad':
        if 'rs' in version_folder:
            wh = version_folder.split('rs')[-1]
            wh_length = len(wh)
            rs_img_size_w, rs_img_size_h = int(wh[:wh_length//2]), int(wh[wh_length//2:])
        else:
            rs_img_size_w = 128
            rs_img_size_h = 32
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'

        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
            
        defect_code_slim = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_sdk_decode = {'ok': 2, 'undersolder': 3, 'pseudosolder': 4}

    elif region == 'singlepad':
        if 'rs' in version_folder:
            wh = version_folder.split('rs')[-1]
            wh_length = len(wh)
            rs_img_size_w, rs_img_size_h = int(wh[:wh_length//2]), int(wh[wh_length//2:])
        else:  
            rs_img_size_w = 64
            rs_img_size_h = 64
        
        print(f"rs_img_size_w = {rs_img_size_w} \n rs_img_size_h = {rs_img_size_h}")
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'

        defect_code_slim = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_sdk_decode = {'ok': 2, 'undersolder': 3, 'pseudosolder': 4}

    elif region == 'pins_part' or region == 'padgroup':
        if '_' in version_folder:
            rs_img_size_w = int(version_folder.split('_')[1])
            rs_img_size_h = int(version_folder.split('_')[2])
#             img_color = version_folder.split('_')[3]
            # img_color = 'rgb'
        else:
            rs_img_size_w, rs_img_size_h = 128, 128
            # img_color = 'rgb'
        rsstr = f'rs{rs_img_size_w}{rs_img_size_h}'

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

    defect_full_decode = {v: k for k, v in defect_sdk_decode.items()}
    n_class = len(defect_code)
    # defect_decode = {i: v for i, v in enumerate(defect_list)}
    defect_decode = {i: k for i, k in enumerate(defect_code.keys())}
    defect_convert = {v: i for i, v in enumerate(defect_code.values())}
    defect_decode.update({-1: 'ng'})
    # load test images
    val_image_pair_data_list = [pd.read_csv(path) for path in val_image_pair_path_list]
    val_image_pair_data_raw = pd.concat(val_image_pair_data_list).reset_index(drop=True)

    if 'defect_dir' in val_image_pair_data_raw.columns:
        val_image_pair_data_raw = val_image_pair_data_raw[val_image_pair_data_raw['defect_dir'] != 'WORD_UNPAIR']

    val_image_pair_data_raw = val_image_pair_data_raw.drop_duplicates(subset=['ref_image', 'insp_image']).reset_index(drop=True)

    if region == 'body' and 'slim' in version_folder:
        val_image_pair_data_raw['insp_y_raw_old'] = list(val_image_pair_data_raw['insp_y_raw'])
        val_image_pair_data_raw['insp_y_raw'] = [3 if y == 1 else y for y in val_image_pair_data_raw['insp_y_raw_old']]  

    val_image_pair_data = val_image_pair_data_raw[
            [y in list(defect_code_slim.values()) for y in val_image_pair_data_raw['insp_y_raw']]]
    val_image_pair_data['insp_y'] = [defect_convert[yraw] for yraw in val_image_pair_data['insp_y_raw']]
    
    # if args.confidence == 'certain':
    val_image_pair_data = val_image_pair_data[
            (val_image_pair_data['confidence'] == 'certain') | (
                    val_image_pair_data['confidence'] == 'unchecked')].copy().reset_index(drop=True)
    
    val_image_pair_data = filter_from_df(val_image_pair_data, args.light_device)
    print(len(val_image_pair_data))
#     val_image_pair_data = val_image_pair_data[:600]
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
    ref_image_label_batches = []
    insp_image_label_batches = []
    binary_label_batchs = []

    n_batches = int(len(val_image_pair_data) / batch_size)

    no_pairs = []
    no_pair_id = []
    if len(val_image_pair_data) < batch_size:
        batch_idices_list = [range(len(val_image_pair_data))]
        n_batches = 1
    else:
        batch_idices_list = split(list(val_image_pair_data.index), n_batches)

    for batch_indices in batch_idices_list:
        
        ref_batch = []
        insp_batch = []
        if calibrate:
            ref_batch_label = []
            insp_batch_label = []
            per_binary_label = []

        for i in batch_indices:
            
            if counter % 1000 == 0:
                print(f'{counter}')
            counter += 1
            val_image_pair_i = val_image_pair_data.iloc[i]

            ref_image_path = os.path.join(test_image_folder, val_image_pair_i['ref_image'])
            insp_image_path = os.path.join(test_image_folder, val_image_pair_i['insp_image'])

            ref_image_name = val_image_pair_i['ref_image'].split('/')[-1]
            insp_image_name = val_image_pair_i['insp_image'].split('/')[-1]
            # scale, normalize and resize test images
            # if random.random() <= sharpness_p:
            #     transform_in = transform
            # else:
            #     transform_in = None

            if fusion:
                if '_rgb_' in ref_image_path:
                    ref_image_path_match = ref_image_path.replace('_rgb_', '_white_')
                    insp_image_path_match = insp_image_path.replace('_rgb_', '_white_')

                    
                elif '_white_' in ref_image_path:
                    ref_image_path_match = ref_image_path.replace('_white_', '_rgb_')
                    insp_image_path_match = insp_image_path.replace('_white_', '_rgb_')

                elif '_RGB_' in ref_image_path:
                    ref_image_path_match = ref_image_path.replace('_RGB_', '_WHITE_')
                    insp_image_path_match = insp_image_path.replace('_RGB_', '_WHITE_')

                    
                elif '_WHITE_' in ref_image_path:
                    ref_image_path_match = ref_image_path.replace('_WHITE_', '_RGB_')
                    insp_image_path_match = insp_image_path.replace('_WHITE_', '_RGB_')
                else:
                    print(f"ref_image_path = {ref_image_path}")
                    print(f"insp_image_path = {insp_image_path}")
                    no_pairs.append((ref_image_path.split(test_image_folder)[-1], insp_image_path.split(test_image_folder)[-1]))
                    no_pair_id.append(i)
                    continue

                if not os.path.exists(ref_image_path_match) or not os.path.exists(insp_image_path_match):
                    no_pairs.append((ref_image_path.split(test_image_folder)[-1], insp_image_path.split(test_image_folder)[-1]))
                    no_pair_id.append(i)

                    continue

                if fusion == 'cat_gray' or fusion == 'merge_rb_G':
                    gray_mean = [0.456]
                    gray_std = [0.224]
                else:
                    gray_mean = None
                    gray_std = None

                ref_image = TransformImageFusion(img_path=ref_image_path, img_type = args.img_type,  rs_img_size_h= rs_img_size_h, gray_mean = gray_mean, gray_std = gray_std,
                                       rs_img_size_w = rs_img_size_w, transform=transform, sharpness_save =sharpness_save, fusion_type=fusion,
                                      ).transform()  # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p
                insp_image = TransformImageFusion(img_path=insp_image_path, img_type = args.img_type, rs_img_size_h=rs_img_size_h, gray_mean = gray_mean, gray_std = gray_std,
                                        rs_img_size_w=rs_img_size_w, transform=transform, sharpness_save =sharpness_save, fusion_type=fusion,
                                        ).transform() # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p

                # ref_image = TransformImage(img_path=ref_image_path, img_type = args.img_type,  rs_img_size_h= rs_img_size_h,
                #                         rs_img_size_w = rs_img_size_w, transform=transform_in, sharpness_save =sharpness_save
                #                         ).transform()  # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p
                # insp_image = TransformImage(img_path=insp_image_path, img_type = args.img_type, rs_img_size_h=rs_img_size_h,
                #                             rs_img_size_w=rs_img_size_w, transform=transform_in, sharpness_save =sharpness_save
                #                             ).transform() # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p

            else:
                ref_image = TransformImage(img_path=ref_image_path, img_type = args.img_type,  rs_img_size_h= rs_img_size_h,
                                        rs_img_size_w = rs_img_size_w, transform=transform, sharpness_save =sharpness_save).transform()  # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p
                insp_image = TransformImage(img_path=insp_image_path, img_type = args.img_type, rs_img_size_h=rs_img_size_h,
                                            rs_img_size_w=rs_img_size_w, transform=transform, sharpness_save =sharpness_save).transform() # lut_path=os.path.join(image_folder, 'merged_annotation','insp.lut.png'), lut_p=lut_p

            # get ground truth labels
            ref_label = val_image_pair_i['ref_y']
            insp_label = val_image_pair_i['insp_y']
            ref_batch.append(ref_image)
            insp_batch.append(insp_image)

            ref_label_list.append(ref_label)
            insp_label_list.append(insp_label)

            if calibrate:
                per_binary_label.append(val_image_pair_i['binary_y'])
                
                ref_batch_label.append(ref_label)
                insp_batch_label.append(insp_label)

            ref_image_name_list.append(ref_image_name)
            insp_image_name_list.append(insp_image_name)

        # print(f"ref_batch = {ref_batch}")
        # print(f"insp_batch = {insp_batch}")
        if len(ref_batch) == 0 or len(insp_batch) == 0:
            continue

        ref_image_batches.append(np.concatenate(ref_batch, axis=0))
        insp_image_batches.append(np.concatenate(insp_batch, axis=0))

        if calibrate:
            ref_image_label_batches.append(np.array(ref_batch_label))
            insp_image_label_batches.append(np.array(insp_batch_label))
            binary_label_batchs.append(np.array(per_binary_label))

    # n_batches = np.max([int(len(ref_image_list) / batch_size), 1])
    no_pairs_df = pd.DataFrame(no_pairs, columns=val_image_pair_res.columns)
    # val_image_pair_res = val_image_pair_res[]

    val_image_pair_res_diff = val_image_pair_res.merge(no_pairs_df, on=list(val_image_pair_res.columns), how='left', indicator=True)
    val_image_pair_res = val_image_pair_res_diff[val_image_pair_res_diff['_merge'] == 'left_only'].drop(columns='_merge')
    binary_label_list = [x for i, x in enumerate(binary_label_list) if i not in no_pair_id]
    binary_labels = np.array(binary_label_list).reshape([-1, 1])
    ref_labels = np.array(ref_label_list).reshape([-1, 1])
    insp_labels = np.array(insp_label_list).reshape([-1, 1])

    if calibrate:
        return n_class, insp_label_list, ref_image_name_list, defect_code, defect_decode, defect_full_decode, val_image_pair_res, binary_labels, ref_labels, \
           insp_labels, ref_image_batches, insp_image_batches, insp_image_name_list, ref_image_label_batches, insp_image_label_batches, binary_label_batchs
    else:
        return n_class, insp_label_list, ref_image_name_list, defect_code, defect_decode, defect_full_decode, val_image_pair_res, binary_labels, ref_labels, \
           insp_labels, ref_image_batches, insp_image_batches, insp_image_name_list

def get_test_all_csvs(image_folder, date,  part_name, valdataset, img_color='rgb'):
    algo_test_csvs = {
        'body':{
                # 广东深圳兰顺现场返回chip件损坏
                'gzls_chip_1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250725_final_white_DA1472_gzls_chip_op_dropoldpairs.csv'),
                'gzls_chip_2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250725_final_white_DA1472_gzls_chip_op_dropoldpairs.csv'),
                'gzls_chip_3' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250725_final_white_DA1472_gzls_chip_op_dropoldpairs_ng_cross_pair_clean.csv'),
                'gzls_chip_4' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250725_final_white_DA1472_gzls_chip_op_dropoldpairs_ok_cross_pair_clean.csv'),

                                                    
                # 深圳利速达错件
                'szlsd_wrong1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250605_final_white_DA1138_szlsd_dropoldpairs.csv'),
                'szlsd_wrong2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250605_final_white_DA1138_szlsd_dropoldpairs.csv'),
                
                # 东莞紫檀山破损
                'dwzts_wrong1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs.csv'),
                'dwzts_wrong2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_transpose.csv'),
                'dwzts_wrong3' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair.csv'),
                'dwzts_wrong4' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair_transpose.csv'),
                'dwzts_wrong5' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ok_cross_pair.csv'),
                'dwzts_wrong6' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ok_insp_cross_pair.csv'),

                'dwzts_wrong7' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs.csv'),
                'dwzts_wrong8' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_insp_ok_cross_pair_up.csv'),
                'dwzts_wrong9' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair_up.csv'),
                'dwzts_wrong10' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair_up_transpose.csv'),
                'dwzts_wrong11' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_ok_cross_pair.csv'),

                # 补充的2DLED
                'dwzts_wrong12' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250526_final_white_DA88888_2d_led_dropoldpairs.csv'),
                'dwzts_wrong13' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250526_final_white_DA88888_2d_led_dropoldpairs.csv'),
                                                       
                                                       

                # 高测破损
                'qb_wrong1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250521_final_white_DA1062_shqb_dropoldpairs.csv'),
                'qb_wrong2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250521_final_white_DA1062_shqb_dropoldpairs.csv'),
               
                # 高测破损
                'gc_wrong1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250520_final_white_DA1057_gc_op_dropoldpairs.csv'),
                'gc_wrong2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250520_final_white_DA1057_gc_op_dropoldpairs.csv'),
                
                # 金赛点软板测试
                'jsd_soft1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250507_final_white_DA978_dropoldpairs.csv'),
                'jsd_soft2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250507_final_white_DA978_dropoldpairs.csv'),


                # 蒙老师黑盒测试
                'blacktestmz' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_pair_body_input_refine_250424.csv'),
                # 全量清理数据
                'body_CONCAT' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'body_CONCAT.csv'),
                'body_OVER_SOLDER' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'body_OVER_SOLDER.csv'),
                'body_SEHUAN_NG' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'body_SEHUAN_NG.csv'),
                'body_SEHUAN_OK' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'body_SEHUAN_OK.csv'),                   
                'body_WORD_PAIR' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'body_WORD_PAIR.csv'), 
                # 聚力创LED错件漏报数据
                'val_image_pair_jlc_path1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250407_final_white_DA855_jlc_dropoldpairs_update6_250418.csv'),
                'val_image_pair_jlc_path2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250407_final_white_DA855_jlc_dropoldpairs_update6_250418.csv'),
                # 硬姐chip误报缺件立碑
                'val_image_pair_yj_path1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250326_final_white_DA820_fp_dropoldpairs_update6_250418.csv'),
                'val_image_pair_yj_path2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250326_final_white_DA820_fp_dropoldpairs_update6_250418.csv'),

                'val_image_pair_796_797_path1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250319_final_white_DA796_797_dropoldpairs_update6_250418.csv'),
                'val_image_pair_796_797_path2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250319_final_white_DA796_797_dropoldpairs_update6_250418.csv'),            
                'val_image_pair_796_797_tg_path1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250319_final_white_DA796_797_dropoldpairs_tg_update6_250418.csv'),


                'val_image_pair_path1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_val_pair_labels_body_merged_withcpmlresorted_model_cleaned_250410_update6_250421.csv'),
                'val_image_pair_path3' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'bbtest_labels_pairs_body_230306v2_withcpresorted_model_cleaned_250410_update6_250418.csv'),
                                                      

                'val_image_pair_path4' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_240328b_finalresorted_model_cleaned_250306_2_update6_250421.csv'),
                                                                                
                'val_image_pair_path5' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_val_pair_labels_body_240328b_finalresorted_model_cleaned_250407_update6_250421.csv'),                                                      

                # debug 文件，未更新，因为没用到
                # 'val_image_pair_path99' : os.path.join(image_folder, 'merged_annotation', date,
                #                                     f'defect_body_pair_input_12664.csv'),
                
                'val_image_pair_path7' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_240819_final_white_model_cleaned_250306_update6_250421.csv'),
                'val_image_pair_path8' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_240819_final_white_model_cleaned_250306_update6_250421.csv')   ,
                'val_image_pair_path9' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_240913_final_white_model_cleaned_250306_update6_250421.csv') ,
                
                'val_image_pair_path10' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_240918_final_white_model_cleaned_250306_update6_250421.csv') ,
                'val_image_pair_path11' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_240919_final_white_model_cleaned_250306_update6_250421.csv') ,
                
                'val_image_pair_path12' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_240926sub_final_white_model_cleaned_250306_update6_250421.csv') ,
                                                      
                'val_image_pair_path100' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'bbtestmz_pair_body_update6_augmented_model_cleaned_250410_2_update6_250421.csv'),
                
                # LED cross pair 241030
                'val_image_pair_path13' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'LED_cross_pairs_ng_test_model_cleaned_250410_update6_250421.csv'),
                'val_image_pair_path14' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'LED_cross_pairs_ok_test_model_cleaned_250306_update6_250421.csv'),
                
                        
                # melf cross pair 241030
                'val_image_pair_path15' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'body_241031_melf_ng_cross_pair_test_model_cleaned_250306_update6_250421.csv'),
                'val_image_pair_path16' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'body_241031_melf_ok_cross_pair_test_model_cleaned_250306_update6_250421.csv'),

                # 17是手动生成的损件数据
                'val_image_pair_path17' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'body_white_paris_final_test_refine2_update6_250418.csv'),

                'val_image_pair_path17_1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'body_white_paris_final_train_refine2_model_cleaned_250306_update6_250418.csv'),

                # DA680body损件， 
                'val_image_pair_path19' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250122_final_white_DA680_dropoldpairs_model_cleaned_250306_2_update6_250418.csv'),
                                                            
        
                # 682是根据标注为连锡的损件批量补充的wrong数据，有加入到训练
                'val_image_pair_path22' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250122_final_white_DA682_dropoldpairs_model_cleaned_250306_2_update6_250418.csv')   ,

                # 685，686全部用于黑盒测试bbtest，数据增强倍数是1，包含所有原始的现场pair，只有少数的增强，
                # 685是高测MIC翻面一个，686是补充的高测MIC翻面三个，686-2是补充的高测MIC ok数据，含部分ng
                'val_image_pair_path20' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250123_final_white_DA685_dropoldpairs_model_cleaned_25423.csv')  ,
                'val_image_pair_path21' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250123_final_white_DA685_dropoldpairs_model_cleaned_25423.csv') ,
                'val_image_pair_path24' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250123_final_white_DA686_dropoldpairs_model_cleaned_25423.csv')  ,
                'val_image_pair_path25' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250123_final_white_DA686_dropoldpairs_model_cleaned_25423.csv')  ,
                'val_image_pair_path26' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250123_final_white_DA686-2_dropoldpairs_model_cleaned_25423.csv')  ,
                'val_image_pair_path27' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250123_final_white_DA686-2_dropoldpairs_model_cleaned_25423.csv') ,
                # 高测DA727 翻面
                'val_image_pair_gaoce_1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250217_final_white_DA727_dropoldpairs_model_cleaned_250306_2_update6_250418.csv')  ,
                                                    
                'val_image_pair_gaoce_2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250217_final_white_DA727_dropoldpairs_model_cleaned_250306_update6_250418.csv') ,

                'val_image_pair_gaoce_3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_body_250218_final_white_DA727_dropoldpairs_model_cleaned_250306_2_update6_250418.csv') ,
                                                    
                'val_image_pair_gaoce_4' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_250218_final_white_DA727_dropoldpairs_model_cleaned_250306_update6_250418.csv') ,
                
                # body 不同料号组合的ng数据,测试模型过拟合程度
                # val_image_pair_aug_ng_1= os.path.join(image_folder, 'merged_annotation', date,
                #                                     f'defect_labels_body_white_pairs_aug_ng_train_size_match.csv')  
                'val_image_pair_aug_ng_2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_labels_body_white_pairs_aug_ng_train_soft_bad_size2_update6_250418.csv') ,
                # val_image_pair_aug_ng_5 = os.path.join(image_folder, 'merged_annotation', date,
                #                                     f'defect_labels_body_white_pairs_aug_ng_test_soft_bad_size.csv')
                
                # val_image_pair_aug_ng_3 = os.path.join(image_folder, 'merged_annotation', date,
                #                                     f'defect_labels_body_white_pairs_aug_ng_train_hard_bad_size.csv') 

                'val_image_pair_aug_ng_4' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_labels_body_white_pairs_aug_ng_test_match_size2_update6_250418.csv') ,
        },
        'padgroup':{
                # 南京焊兆连锡漏保(仅白图缺陷特征明显, rgb的ng很不明显)
                'njzh_rgbwhite1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                'njzh_rgbwhite2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv'),

                'njzh_rgbwhite3': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv'),

                # 河南郑州卓威电子LED误报连锡
                'hnzzzw_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs.csv'),
                'hnzzzw_rgb2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs.csv'),

                # 河南郑州卓威电子LED误报连锡
                'hnzzzw_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs.csv'),
                'hnzzzw_rgb2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs.csv'),
                'hnzzzw_white1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs.csv'),
                'hnzzzw_white2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs.csv'),

                # 河南郑州卓威电子LED误报连锡(TG)
                'hnzzzw_rgb_TG1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs_TG.csv'),
                'hnzzzw_rgb_TG2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs_TG.csv'),
                'hnzzzw_white_TG1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs_TG.csv'),
                'hnzzzw_white_TG2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs_TG.csv'),


                # 深圳长城白图漏报（rgb图太不明显,将白图加入rgb图）,暂时当做特供数据处理，待训练后看效果
                'szcc_lx_rgb_op1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_250711_final_rgb_DA1397_szcc_lx_op_dropoldpairs.csv'),
                'szcc_lx_rgb_op2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250711_final_rgb_DA1397_szcc_lx_op_dropoldpairs.csv'),
                'szcc_lx_rgb_mask_op1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_250711_final_rgb_mask_DA1397_szcc_lx_op_dropoldpairs.csv'),
                'szcc_lx_rgb_mask_op2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250711_final_rgb_mask_DA1397_szcc_lx_op_dropoldpairs.csv'),

                'szcc_lx_white_op1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_250711_final_white_DA1397_szcc_lx_op_dropoldpairs.csv'),
                'szcc_lx_white_op2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250711_final_white_DA1397_szcc_lx_op_dropoldpairs.csv'),
                'szcc_lx_white_mask_op1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_250711_final_white_mask_DA1397_szcc_lx_op_dropoldpairs.csv'),
                'szcc_lx_white_mask_op2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250711_final_white_mask_DA1397_szcc_lx_op_dropoldpairs.csv'),


                # 深圳诚而信误报数据(本体过多)
                'szcex_rgb_clean_fp1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250708_final_rgb_DA1364_cex_led_op_dropoldpairs.csv'),
                'szcex_rgb_clean_fp2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250708_final_rgb_DA1364_cex_led_op_dropoldpairs_ok_cross_pair_clean_up.csv'),

                'szcex_white_clean_fp1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250708_final_white_DA1364_cex_led_op_dropoldpairs.csv'),
                'szcex_white_clean_fp2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_250708_final_white_DA1364_cex_led_op_dropoldpairs_ok_cross_pair_clean_up.csv'),


                # 3D连锡补充数据
                '3D_rgb_supply1': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250707_final_rgb_DA1339_3D_short_dropoldpairs.csv'),
                '3D_rgb_mask_supply1': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250707_final_rgb_mask_DA1339_3D_short_dropoldpairs.csv'),
                '3D_white_supply1': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250707_final_white_DA1339_3D_short_dropoldpairs.csv'),
                '3D_white_mask_supply1': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250707_final_white_mask_DA1339_3D_short_dropoldpairs.csv'),

                # 硬姐连锡漏报
                'yj_lx_rgb_op1': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250703_final_rgb_DA1321_yj_lxop_dropoldpairs_ok_cross_pair_clean.csv'),
                'yj_lx_rgb_op2': os.path.join(image_folder, 'merged_annotation', date,
                                                       f'aug_train_pair_labels_padgroup_250703_final_rgb_DA1321_yj_lxop_dropoldpairs.csv'),
                'yj_lx_rgb_op3': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250703_final_rgb_DA1321_yj_lxop_dropoldpairs_ok_cross_pair_clean.csv'),
                'yj_lx_rgb_op4': os.path.join(image_folder, 'merged_annotation', date,
                                                       f'aug_test_pair_labels_padgroup_250703_final_rgb_DA1321_yj_lxop_dropoldpairs.csv'),
                'yj_lx_rgb_mask_op1': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250703_final_rgb_mask_DA1321_yj_lxop_dropoldpairs.csv'),
                'yj_lx_rgb_mask_op2': os.path.join(image_folder, 'merged_annotation', date,
                                                       f'aug_test_pair_labels_padgroup_250703_final_rgb_mask_DA1321_yj_lxop_dropoldpairs.csv'),

                'yj_lx_white_op1': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250703_final_white_DA1321_yj_lxop_dropoldpairs_ok_cross_pair_clean.csv'),
                'yj_lx_white_op2': os.path.join(image_folder, 'merged_annotation', date,
                                                       f'aug_train_pair_labels_padgroup_250703_final_white_DA1321_yj_lxop_dropoldpairs.csv'),
                'yj_lx_white_op3': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250703_final_white_DA1321_yj_lxop_dropoldpairs_ok_cross_pair_clean.csv'),
                'yj_lx_white_op4': os.path.join(image_folder, 'merged_annotation', date,
                                                       f'aug_test_pair_labels_padgroup_250703_final_white_DA1321_yj_lxop_dropoldpairs.csv'),
                'yj_lx_white_mask_op1': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250703_final_white_mask_DA1321_yj_lxop_dropoldpairs.csv'),
                'yj_lx_white_mask_op2': os.path.join(image_folder, 'merged_annotation', date,
                                                       f'aug_test_pair_labels_padgroup_250703_final_white_mask_DA1321_yj_lxop_dropoldpairs.csv'),
          

                # 深圳鹏ju漏报截图
                'szpj': os.path.join(image_folder, 'merged_annotation', date,
                                f'szpj.csv'),
                # 郑州装联漏报
                'zzzl_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                f'aug_test_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl_dropoldpairs.csv'),
                'zzzl_rgb2': os.path.join(image_folder, 'merged_annotation', date,
                                f'aug_test_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl2_dropoldpairs.csv'),                                        
                # 郑州装联漏报测试集中ng，ok数量较少，扩充一下
                'zzzl_rgb3': os.path.join(image_folder, 'merged_annotation', date,
                                f'aug_train_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl_dropoldpairs_ng_cross_pair.csv'),
                'zzzl_rgb4': os.path.join(image_folder, 'merged_annotation', date,
                                f'aug_train_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl_dropoldpairs_ok_cross_pair_up.csv'),

                # 郑州装联漏报
                'zzzl_rgb_mask1': os.path.join(image_folder, 'merged_annotation', date,
                                                       f'aug_test_pair_labels_padgroup_250625_final_rgb_mask_DA1272_hnzl_dropoldpairs.csv'),
                'zzzl_rgb_mask2': os.path.join(image_folder, 'merged_annotation', date,
                                                       f'aug_test_pair_labels_padgroup_250625_final_rgb_mask_DA1272_hnzl2_dropoldpairs.csv'),

                # 郑州装联漏报
                'zzzl_white1': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250625_final_white_DA1272_hnzl_dropoldpairs.csv'),
                'zzzl_white2': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250625_final_white_DA1272_hnzl2_dropoldpairs.csv'),
                'zzzl_white3': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250625_final_white_DA1272_hnzl_dropoldpairs_ng_cross_pair.csv'),

                'zzzl_white4': os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250625_final_white_DA1272_hnzl_dropoldpairs_ok_cross_pair_up.csv'),
                # 郑州装联漏报
                'zzzl_white_mask1': os.path.join(image_folder, 'merged_annotation', date,
                                f'aug_test_pair_labels_padgroup_250625_final_white_mask_DA1272_hnzl_dropoldpairs.csv'),
                'zzzl_white_mask2': os.path.join(image_folder, 'merged_annotation', date,
                                f'aug_test_pair_labels_padgroup_250625_final_white_mask_DA1272_hnzl2_dropoldpairs.csv'),                           

                # 江西蓝之洋漏报误报
                'jxlzy_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250610_final_rgb_DA1167_1168_jxlzy_2_dropoldpairs.csv'),
                'jxlzy_rgb2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250610_final_rgb_DA1167_1168_jxlzy_2_dropoldpairs.csv'),                                                
                
                'jxlzy_white1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250610_final_white_DA1167_1168_jxlzy_2_dropoldpairs.csv'),
                'jxlzy_white2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250610_final_white_DA1167_1168_jxlzy_2_dropoldpairs.csv'),                                                
                
                'jxlzy_mask_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250610_final_rgb_mask_DA1167_1168_jxlzy_2_dropoldpairs.csv'),
                'jxlzy_mask_rgb2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250610_final_rgb_mask_DA1167_1168_jxlzy_2_dropoldpairs.csv'),                                                
                'jxlzy_mask_white1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250610_final_white_mask_DA1167_1168_jxlzy_2_dropoldpairs.csv'),
                'jxlzy_mask_white2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250610_final_white_mask_DA1167_1168_jxlzy_2_dropoldpairs.csv'),                                                

                'jxlzy_TG_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250610_final_rgb_DA1167_1168_jxlzy_2_dropoldpairs_TG.csv'),
                'jxlzy_TG_rgb2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250610_final_rgb_DA1167_1168_jxlzy_2_dropoldpairs_TG.csv'),                                                
                
                'jxlzy_TG_white1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250610_final_white_DA1167_1168_jxlzy_2_dropoldpairs_TG.csv'),
                'jxlzy_TG_white2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250610_final_white_DA1167_1168_jxlzy_2_dropoldpairs_TG.csv'),                                                
                             
                # 3D数据
                '3D_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250423_final_rgb_DA920_3D_dropoldpairs_clean.csv'),
                '3D_mask_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250423_final_rgb_mask_DA920_3D_dropoldpairs_clean.csv'),                                                
                '3D_white1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250423_final_white_DA920_3D_dropoldpairs_clean.csv'),
                '3D_mask_white1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250423_final_white_mask_DA920_3D_dropoldpairs_clean.csv'),                                                               
                # 韶关嘉立创漏报（非常轻微的连锡）
                'sgjlc_rgb_250422': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs.csv'),
                'sgjlc_white_250422': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250422_final_white_DA922_sgjlc2_dropoldpairs.csv'),                                                
                
                'extrem_rgb_ng': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'padgroup_extrem_rgb_ng.csv'),
                'extrem_white_ng': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'padgroup_extrem_white_ng.csv'),
                # 欣润连锡误报
                'xr_rgb3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250414_final_rgb_DA887_szxr_dropoldpairs.csv'),
                'xr_rgb4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250414_final_rgb_DA887_szxr_dropoldpairs.csv'),
                'xr_white3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250414_final_white_DA887_szxr_dropoldpairs.csv'),
                'xr_white4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250414_final_white_DA887_szxr_dropoldpairs.csv'),

                # 欣润连锡漏报
                'xr_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs.csv'),
                'xr_rgb2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs.csv'),
                'xr_white1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250411_final_white_DA873_szxr_dropoldpairs.csv'),
                'xr_white2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250411_final_white_DA873_szxr_dropoldpairs.csv'),
                
                # 德洲连锡漏报,原始数据作为测试集， cross pair用于训练
                'dz_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250409_final_rgb_DA857_zz_add_ok_dropoldpairs.csv'),
                'dz_rgb2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250409_final_rgb_DA857_zz_add_ok_dropoldpairs.csv'),
                'dz_white1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250409_final_white_DA857_zz_add_ok_dropoldpairs.csv'),
                'dz_white2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250409_final_white_DA857_zz_add_ok_dropoldpairs.csv'),
                
                # 德洲连锡漏报
                'padGroup_5_rgb': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'padGroup_5_rgb.csv'),
                'padGroup_5_white': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'padGroup_5_white.csv'),
               # 深圳仁盈连锡误报,扩大body
                'val_image_pair_ry_syh_rgb_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv'),
                'val_image_pair_ry_syh_rgb_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv'),
                'val_image_pair_ry_syh_white_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250403_final_white_DA852_ry_syh_dropoldpairs.csv'),
                'val_image_pair_ry_syh_white_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250403_final_white_DA852_ry_syh_dropoldpairs.csv'),
               # 深圳仁盈连锡误报
                'val_image_pair_ry_rgb_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250403_final_rgb_DA852_ry_dropoldpairs.csv'),
                'val_image_pair_ry_rgb_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250403_final_rgb_DA852_ry_dropoldpairs.csv'),
                'val_image_pair_ry_white_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250403_final_white_DA852_ry_dropoldpairs.csv'),
                'val_image_pair_ry_white_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250403_final_white_DA852_ry_dropoldpairs.csv'),

               # 易创亿USB连锡漏报
                'val_image_pair_ycy_debug_rgb_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_debug_SMTAOITS-1637_update.csv'),
                'val_image_pair_ycy_debug_rgb_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_debug_SMTAOITS-1638_update.csv'),
                'val_image_pair_ycy_rgb_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250324_final_rgb_DA815_op_dropoldpairs.csv'),
                'val_image_pair_ycy_rgb_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250324_final_rgb_DA815_op_dropoldpairs_update_250514.csv'),
                'val_image_pair_ycy_white_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250324_final_white_DA815_op_dropoldpairs.csv'),
                'val_image_pair_ycy_white_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250324_final_white_DA815_op_dropoldpairs.csv'),               
               # 澜悦漏报数据DA807
                'val_image_pair_yl_rgb_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250321_final_rgb_DA807_op_dropoldpairs.csv'),
                'val_image_pair_yl_rgb_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250321_final_rgb_DA807_op_dropoldpairs.csv'),
                'val_image_pair_yl_white_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250321_final_white_DA807_op_dropoldpairs.csv'),
                'val_image_pair_yl_white_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250321_final_white_DA807_op_dropoldpairs.csv'),

               # 金赛点误报数据DA786
                'val_image_pair_jsd_rgb_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv'),
                'val_image_pair_jsd_rgb_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv'),
                'val_image_pair_jsd_white_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_padgroup_250317_final_white_DA786_solder_shortage_fp_dropoldpairs.csv'),
                'val_image_pair_jsd_white_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_padgroup_250317_final_white_DA786_solder_shortage_fp_dropoldpairs.csv'),


                'val_image_pair_path1_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_v090nonpaired_rgb_final3_up_250707.csv'),
                'val_image_pair_path2_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_v090merged_rgb_final3_up_250707.csv'),
                'val_image_pair_path3_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_240407_final_rgb_final3_up_250707.csv'),
                'val_image_pair_path4_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240407_final_rgb_final3_up_250707.csv'),

                'val_image_pair_path7_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'bbtest_labels_pairs_padgroup_230306v2_rgb_final3_up_250707.csv'),
                
                'val_image_pair_path8_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_240417_final_RGB_up_250707.csv'),
                'val_image_pair_path9_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_240418_final_RGB_up_250707.csv'),
                'val_image_pair_path10_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_240419_final_RGB_up_250707.csv'),
        #         val_image_pair_path10 : os.path.join(image_folder, 'merged_annotation', date,
        #                      f'aug_train_pair_labels_padgroup_240417_final_WHITE.csv')
        #         val_image_pair_path11 : os.path.join(image_folder, 'merged_annotation', date,
        #                      f'aug_train_selfpair_labels_padgroup_240417_final_white.csv')
                'val_image_pair_path12_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240424_final_RGB_up_250707.csv'),
                'val_image_pair_path13_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240428_final_RGB_up_250707.csv'),
                'val_image_pair_path14_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240429_final_RGB_up_250707.csv'),
                
                'val_image_pair_path15_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_240419masked_final_RGB_up_250707.csv'),
                'val_image_pair_path16_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240419masked_final_RGB_up_250707.csv'),
                'val_image_pair_path17_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240424_final_RGB_mask_up_250707.csv'),
                'val_image_pair_path18_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240428_final_RGB_mask_up_250707.csv'),
                'val_image_pair_path19_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240429_final_RGB_mask_up_250707.csv'),
                'val_image_pair_path20_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'jira_test_pair_labels_padgroup_240429_up_250707.csv'),
                #debug数据，不加入测试
                # 'val_image_pair_path21' : os.path.join(image_folder, 'merged_annotation', date,
                #             f'aug_debug_pair_labels_padgroup_240627_aug.csv') ,
                
                
                'val_image_pair_path22_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_240715_final_rgb_mask_up_250707.csv'),

                'val_image_pair_path23_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_240715_final_RGB_up_250707.csv') ,
                

                'val_image_pair_path24_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_240701_final_RGB_up_250707.csv') ,
                'val_image_pair_path25_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_240701_final_RGB_mask_up_250707.csv') ,
                
                'val_image_pair_path26_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241023_final_rgb_up_250707.csv') ,
                'val_image_pair_path27_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241023_final_rgb_mask_up_250707.csv') ,

                # padgroup241101
                'val_image_pair_path28_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241101_final_rgb_up_250707.csv') ,
                'val_image_pair_path29_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241101_final_rgb_mask_model_cleaned_graphite_uncertain.csv') ,
                'val_image_pair_path30_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241101_final_rgb_26MHZ_up_250707.csv') ,
                'val_image_pair_path31_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241101_final_rgb_mask_26MHZ_model_cleaned_graphite_uncertain.csv') ,
                'val_image_pair_path32_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241202_final_rgb_DA512_update_241206_up_250707.csv') ,
                
                # 33 34算进了 bbtest
                'val_image_pair_path_rgb33' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_241216_final_rgb_DA1101_up_250707.csv') ,
                'val_image_pair_path_rgb34' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_241216_final_rgb_mask_DA1101.csv') ,
                
                'val_image_pair_path_rgb35' : os.path.join(image_folder, 'merged_annotation', date,
                                f'aug_test_pair_labels_padgroup_241224_final_rgb_DA575and577_drop.csv'),
                'val_image_pair_path_rgb36' : os.path.join(image_folder, 'merged_annotation', date,
                                f'aug_test_pair_labels_padgroup_241224_final_rgb_mask_DA575and577_drop.csv'),
                
                'val_image_pair_path_rgb37' : os.path.join(image_folder, 'merged_annotation', date,
                                                f'aug_test_pair_labels_padgroup_250115_final_rgb_DA656_dropoldpairs.csv') ,
                'val_image_pair_path_rgb38' : os.path.join(image_folder, 'merged_annotation', date,
                                                f'aug_test_pair_labels_padgroup_250118_final_rgb_DA663-670_dropoldpairs.csv'),
                'val_image_pair_path_rgb39' : os.path.join(image_folder, 'merged_annotation', date,
                                                f'aug_test_pair_labels_padgroup_250115_final_rgb_mask_DA656_dropoldpairs.csv'),
                'val_image_pair_path_rgb40' : os.path.join(image_folder, 'merged_annotation', date,
                                                f'aug_test_pair_labels_padgroup_250118_final_rgb_mask_DA663-670_dropoldpairs.csv')  ,

                # 石墨特例
                'val_image_pair_path_rgb400' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'only_graphite.csv')  ,
                
                # white
                'val_image_pair_path5_white' :os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_240407_final_white_final3_model_cleaned_graphite_uncertain.csv'),
                'val_image_pair_path6_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240407_final_white_final3_model_cleaned_graphite_uncertain.csv'),
                'val_image_pair_path33_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_ori_pair_labels_padgroup_241023_final_white_model_cleaned_graphite_uncertain.csv') ,
                'val_image_pair_path34_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240407_final_white_final3_model_cleaned_graphite_uncertain.csv') ,
                'val_image_pair_path35_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240424_final_WHITE_model_cleaned_graphite_uncertain.csv') ,
                'val_image_pair_path36_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240428_final_WHITE_model_cleaned_graphite_uncertain.csv') ,
            
                'val_image_pair_path37_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_240429_final_WHITE_model_cleaned_graphite_uncertain.csv')    ,
                    
                'val_image_pair_path43_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_240407_final_white_final3_model_cleaned_graphite_uncertain.csv')   ,
                
                'val_image_pair_path44_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_240417_final_WHITE_model_cleaned_graphite_uncertain.csv') ,
                
                'val_image_pair_path45_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_240418_final_WHITE_model_cleaned_graphite_uncertain.csv') ,
                
                'val_image_pair_path38_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241023_final_white_update_241109_model_cleaned_graphite_uncertain.csv') ,
                'val_image_pair_path39_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241101_final_white_26MHZ_update_241109_model_cleaned_graphite_uncertain.csv') ,
                'val_image_pair_path40_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241101_final_white_update_241109_model_cleaned_graphite_uncertain.csv') ,
                'val_image_pair_path46_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_val_pair_labels_padgroup_241101_final_white_update_241109_model_cleaned_graphite_uncertain.csv') ,
                'val_image_pair_path42_white' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241202_final_white_DA512_update_241206_model_cleaned_graphite_uncertain.csv') ,
                
                
                # 47, 48算进了bbtest
                'val_image_pair_path_white47' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_241216_final_white_DA1101.csv') ,
                'val_image_pair_path_white48' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_padgroup_241216_final_white_mask_DA1101.csv') ,
                'val_image_pair_path_white49' : os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241224_final_white_DA575and577_drop.csv'),
                'val_image_pair_path_white50' :  os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_padgroup_241224_final_white_mask_DA575and577_drop.csv')  ,

                'val_image_pair_path_white51' : os.path.join(image_folder, 'merged_annotation', date,
                                    f'aug_test_pair_labels_padgroup_250115_final_white_DA656_dropoldpairs.csv'),
                'val_image_pair_path_white52' : os.path.join(image_folder, 'merged_annotation', date,
                                f'aug_test_pair_labels_padgroup_250118_final_white_DA663-670_dropoldpairs.csv'),
                'val_image_pair_path_white53' : os.path.join(image_folder, 'merged_annotation', date,
                                    f'aug_test_pair_labels_padgroup_250115_final_white_mask_DA656_dropoldpairs.csv') ,
                'val_image_pair_path_white54' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250118_final_white_mask_DA663-670_dropoldpairs.csv'),

                # 临时测试数据DA679高测连锡漏报, 1,3test加入黑盒测试bbtest，2，4不可以加入训练
                'val_image_pair_path_gaoce1_rgb' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250121_final_rgb_DA679_dropoldpairs.csv'),

                'val_image_pair_path_gaoce3_white' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250121_final_white_DA679_dropoldpairs.csv'),

                # DA718 LED1.14版本误报, 临时测试
                'val_image_pair_path_rgb_led1' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250214_final_rgb_DA718_dropoldpairs.csv'),
                'val_image_pair_path_rgb_led2' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250214_final_rgb_DA718_dropoldpairs.csv'),
                'val_image_pair_path_white_led1' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250214_final_white_DA718_dropoldpairs.csv'),
                'val_image_pair_path_white_led2' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250214_final_white_DA718_dropoldpairs.csv'),
                
                # 昆山和辉瑞达特供数据筛选后的通用训练测试集
                'val_image_pair_path_ks_hrd_general_rgb1' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_clean.csv'),
                'val_image_pair_path_ks_hrd_general_rgb2' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_two_pad_short_clean.csv')                                   ,
            
                'val_image_pair_path_ks_hrd_general_rgb3' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_two_pad_clean.csv'),

                'val_image_pair_path_ks_hrd_general_white1' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_clean.csv'),
                'val_image_pair_path_ks_hrd_general_white2' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_two_pad_short_clean.csv')                                   ,
            
                'val_image_pair_path_ks_hrd_general_white3' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_two_pad_clean.csv'),

                # DA757深圳永迦误报原始数据
                'val_image_pair_path_yj_rgb1' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250304_final_rgb_DA757_jira1575_dropoldpairs.csv'),
                'val_image_pair_path_yj_rgb2' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250304_final_rgb_DA757_jira1575_dropoldpairs.csv'),
                'val_image_pair_path_yj_white1' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_test_pair_labels_padgroup_250304_final_white_DA757_jira1575_dropoldpairs.csv'),
                'val_image_pair_path_yj_white2' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250304_final_white_DA757_jira1575_dropoldpairs.csv')     ,

                'val_image_pair_path_szzn_rgb1' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250311_final_rgb_DA769_dropoldpairs.csv'),
                'val_image_pair_path_szzn_white1' : os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_train_pair_labels_padgroup_250311_final_white_DA769_dropoldpairs.csv'),
        },
        'singlepinpad':{ 
                # mhdxh_250827
                'mhdxh_250827_1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs.csv'),
                'mhdxh_250827_2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                'mhdxh_250827_3': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_ng_corss_pair_up.csv'),
                'mhdxh_250827_4': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_ok_corss_pair_up2_250828.csv'),

                'mhdxh_250827_TG_1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_TG.csv'),
                'mhdxh_250827_TG_2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_TG_ok_corss_pair_up.csv'),


                # 福建泉州智领虚焊
                'fzqzzl_xh1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs.csv'),
                'fzqzzl_xh2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_ok_corss_pair_up.csv'),

                # 深圳钦盛伟源原始数据（TG)
                'szqswy_ori1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'defect_labels_singlepinpad_rgb_pairs_split_test.csv'),
                
            
                # 广州华创冷焊
                'gzhc_lh1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs.csv'),
                'gzhc_lh2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs_ok_corss_pair_up.csv'),
                'gzhc_lh3': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs.csv'),
                'gzhc_lh4': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs_ok_corss_pair_up.csv'),
                # 天津信天冷焊
                'tjxt_cold_solder1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs.csv'),
                'tjxt_cold_solder2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs.csv'),

                'tjxt_cold_solder3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs_ok_cross_pair_up_cp5.csv'),
                'tjxt_cold_solder4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs_ng_cross_pair_up_cp5.csv'),
                'tjxt_cold_solder5': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs_ng_cross_pair_up.csv'),
                'tjxt_cold_solder6': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs_ok_cross_pair_up.csv'),
                                
                
                # 永林冷焊数据
                'yl_cold_solder1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_up_250522.csv'),
                'yl_cold_solder2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up_up_250522.csv'),
                'yl_cold_solder3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv'),
                'yl_cold_solder4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_ok_cross_pair_up.csv'),

                'yl_cold_solder5': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs.csv'),
                'yl_cold_solder6': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up.csv'),
                'yl_cold_solder7': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv'),
                'yl_cold_solder8': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_ok_cross_pair_up.csv'),

                # 南京兆焊
                'njzh1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                'njzh2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv'),
                # 广州华创漏报
                'gzhc1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs.csv'),
                'gzhc2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs.csv'),
                'gzhc3': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs_ok_corss_pair_up.csv'),

                'gzhc_TG1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs_TG_ok_corss_pair.csv'),
                'gzhc_TG2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs_TG.csv'),            



                # 卓威误报特供
                'zwlb_clean1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_clean.csv'),
                'zwlb_clean2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_clean.csv'),
                'zwlb_clean3': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_ok_corss_pair_up.csv'),
                'zwlb_clean4': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_ok_corss_pair_up.csv'),


                'zwlb_TG1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs.csv'),
                'zwlb_TG2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs.csv'),



                # 南京淼龙虚焊漏报
                'njml_clean1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_clean_ok_corss_pair_up.csv'),
                'njml_clean2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_clean.csv'),
                # 南京淼龙OK特供
                'njml_ng_soft1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ng_soft.csv'),
                'njml_ng_soft2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ng_soft.csv'),
                # 南京淼龙OK特供
                'njml_ng_hard1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_train_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ng_hard.csv'),
                'njml_ng_hard2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ng_hard.csv'),


                # 深圳明弘达返回虚焊未检出
                'szmhd_xh_op1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs.csv'),
                'szmhd_xh_op2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_ok_cross_pair_clean_up.csv'),
                'szmhd_xh_op_NG': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_TG_ng_cross_pair_clean.csv'),
                'szmhd_xh_op_TG': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_TG.csv'),

                # 江苏昆山丘钛翘脚虚焊漏报
                'jsks_xh_op1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250707_final_rgb_DA1341_jxks_xh_op_dropoldpairs_clean.csv'),
                'jsks_xh_op2': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250707_final_rgb_DA1341_jxks_xh_op_dropoldpairs_clean_ng_cross_pair_clean_up.csv'),
                'jsks_xh_TG_op1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250707_final_rgb_DA1341_jxks_xh_op_dropoldpairs_TG.csv'),
                'jsks_xh_supply_op1': os.path.join(image_folder, 'merged_annotation', date,
                                        f'aug_test_pair_labels_singlepinpad_250710_final_rgb_DA1388_jsks_xh_op_dropoldpairs.csv'),
                # 辉瑞达补充特供数据614,532
                'hrd_da614532_1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_singlepinpad_250708_final_rgb_DA1356_hrd614532_op_dropoldpairs.csv'),
                'hrd_da614532_2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_train_pair_labels_singlepinpad_250708_final_rgb_DA1356_hrd614532_op_dropoldpairs.csv'),    

                # 辉瑞达原始补充特供数据614,532
                'hrd_da614532_ori_1': os.path.join(image_folder, 'merged_annotation', date,
                            f'defect_labels_singlepinpad_rgb_pairs_split_hrd_da614532_ori_train.csv'),
                'hrd_da614532_ori_2': os.path.join(image_folder, 'merged_annotation', date,
                            f'defect_labels_singlepinpad_rgb_pairs_split_hrd_da614532_ori_test.csv'),  

                # 辉瑞达新旧数据整理增强版（斌哥，老板）
                'hrd_processed_by_bgjc_aug': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_singlepinpad_250701_final_rgb_hrd_DA_ALL_dropoldpairs_bg_jc_aug.csv'),
                'hrd_processed_by_bgjc_ori1': os.path.join(image_folder, 'merged_annotation', date,
                            f'defect_labels_singlepinpad_rgb_pairs_split_jc_bg_ori_test_sample_ng_up.csv'),
                'hrd_processed_by_bgjc_ori2': os.path.join(image_folder, 'merged_annotation', date,
                            f'defect_labels_singlepinpad_rgb_pairs_split_jc_bg_ori_train_add_ng_up.csv'),
                'hrd_processed_by_jm_robin1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_singlepinpad_250701_final_rgb_DA_ALL_OLD_dropoldpairs_jm_robin.csv'),
                'hrd_processed_by_jm_robin2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_singlepinpad_250703_final_rgb_DA1310_hrd_supply_dropoldpairs.csv'),
                     

                # 深圳达人高科翘脚漏报
                'szgkd1': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_singlepinpad_250627_final_rgb_DA1292_szdrgk_dropoldpairs.csv'),
                'szgkd2': os.path.join(image_folder, 'merged_annotation', date,
                            f'aug_test_pair_labels_singlepinpad_250627_final_rgb_DA1292_szdrgk_dropoldpairs_ok_cross_pair_up.csv'),
                                        
                # 澜悦翘脚漏报
                'ly_qj_250620_op1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250620_final_rgb_DA1241_ly_qj_dropoldpairs.csv'),
                'ly_qj_250620_op2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250620_final_rgb_DA1241_ly_qj_dropoldpairs.csv'),  
                
                # 辉瑞达翘脚漏报
                'hrd_led_qj1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj1_dropoldpairs.csv'),
                'hrd_led_qj2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj1_dropoldpairs.csv'),  
                
                # jxlzy 江西蓝之洋特供数据
                'jxlzy_base1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori.csv'),
                'jxlzy_base2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori.csv'),  

                'jxlzy1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250612_final_rgb_DA1183_jxlzy_solder_dropoldpairs_ori.csv'),
                'jxlzy2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250612_final_rgb_DA1183_jxlzy_solder_dropoldpairs_ori.csv'),  
                'jxlzyup1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250612_final_rgb_DA1183_jxlzy_solder_dropoldpairs_ori_update.csv'),
                'jxlzyup2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250612_final_rgb_DA1183_jxlzy_solder_dropoldpairs_ori_update.csv'),  
                
                'jxlzy_confirm1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori.csv'),
                'jxlzy_confirm2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori.csv'),  
                
                                                 
            
                # 深圳利速达特供数据
                'szlsd1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250605_final_rgb_DA1138_szlsd_dropoldpairs.csv'),
                'szlsd2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250605_final_rgb_DA1138_szlsd_dropoldpairs_ori.csv'),  
                
                # 惠州光弘
                'hzgh1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs.csv'),
                'hzgh2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs.csv'),  
                'hzgh3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_soft_TG_ok_cross_pair_up.csv'),
                'hzgh4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_hard_TG_ok_cross_pair_up.csv'),                                              
                

                # 深圳富洛翘脚
                'szfl_qj1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250522_final_rgb_DA1077_fl_dropoldpairs.csv'),
                'szfl_qj2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250522_final_rgb_DA1077_fl_dropoldpairs.csv'),

                'szfl_qj3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250527_final_rgb_DA1106_szfl_op_dropoldpairs.csv'),
                'szfl_qj4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250527_final_rgb_DA1106_szfl_op_dropoldpairs.csv'),
                   
        
                # 炉前炉后
                'bp_stove_clean1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs.csv'),
                'bp_stove_clean2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_ok_cross_pair_ok_cross_pair_up.csv'),
                'bp_stove_soft1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft.csv'),
                'bp_stove_soft2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft_ok_cross_pair_up.csv'),
                

                'yl_cold_solder_TG1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_TG.csv'),

                # 3D 数据
                '3D_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250423_final_rgb_DA920_3D_dropoldpairs.csv'),
                                                
                # 金赛点软板错位
                'jsd_soft1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250507_final_rgb_DA978_dropoldpairs.csv'),
                'jsd_soft2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250507_final_rgb_DA978_dropoldpairs.csv'), 

                # 嘉立创3D漏报
                'sgjlc1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250421_final_rgb_DA915_sgjlc_dropoldpairs_up_250522.csv'),
                'sgjlc2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250421_final_rgb_DA915_sgjlc_dropoldpairs_up_250522.csv'), 
                'gznh1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250421_final_rgb_DA917_gznh_dropoldpairs_up_250522.csv'), 
                'gznh2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250421_final_rgb_DA917_gznh_dropoldpairs_ng_cross_pair_up_250522.csv'), 
                'gznh3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250421_final_rgb_DA917_gznh_dropoldpairs_ng_cross_pairs.csv'),  
                # 郑州众智误报虚焊,特供数据
                'val_image_pair_zz_fp_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250409_final_rgb_DA857_zz_dropoldpairs.csv'),
                'val_image_pair_zz_fp_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250409_final_rgb_DA857_zz_dropoldpairs.csv'), 
                # 辉瑞大偏移漏报
                'val_image_pair_hrd_op_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250324_final_rgb_DA816_op_dropoldpairs_update_250630.csv'),
                'val_image_pair_rs_op_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250319_final_rgb_DA795_rs_op_dropoldpairs_up_250522.csv'),
                'val_image_pair_rs_op_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250319_final_rgb_DA795_rs_op_dropoldpairs_up_250522.csv'),

                'val_image_pair_jsd_fp_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250317_final_rgb_DA780_jsd_fp_dropoldpairs.csv'),
                'val_image_pair_jsd_fp_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_250317_final_rgb_DA780_jsd_fp_dropoldpairs.csv'),

                'val_image_pair_path99': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'bbtestmz_pair_pinpad_input_update_model_cleaned.csv'),
                'val_image_pair_path100' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'bbtestmz_pair_pinpad_updateaugmented_model_cleaned_update_250630.csv'),
                'val_image_pair_path1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_val_pair_labels_singlepinpad_merged_update_250630.csv'),
                'val_image_pair_path2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'bbtest_labels_pairs_singlepinpad_230306v2_up_250623.csv'),
                'val_image_pair_path3' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_val_pair_labels_singlepinpad_240329_final_update_250630.csv'),
                'val_image_pair_path4' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_240329_final_update_250630.csv'),
                'val_image_pair_path5' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'jiraissues_pair_labels_singlepinpad_240314_final_update_250630.csv'),
        #         val_image_pair_path6 : os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'test_pair_labels_singlepinpad_240403debug_final.csv')
                # val_image_pair_path6 : os.path.join(image_folder, 'merged_annotation', 'debug',
                #                                     f'debug_singlepinpad_240816_update.csv')
                'val_image_pair_path7' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'test_pair_labels_singlepinpad_240404debug_final_update_250630.csv'),
                
                
                'val_image_pair_path8' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_240424_final_RGB_update_250630.csv'),
                'val_image_pair_path9' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_240428_final_RGB_update_250630.csv'),
                'val_image_pair_path10' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_240429_final_RGB_update_250630.csv'),
                'val_image_pair_path11' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'jira_test_pair_labels_singlepinpad_240429_up_250522.csv'),
                'val_image_pair_path12' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_240716_final_RGB_update_250630.csv'),
                'val_image_pair_path14' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_240913_final_rgb_update_250630.csv'),

                'val_image_pair_path15' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_update_250630.csv'),

                'val_image_pair_path16' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_240919_final_rgb_ng_update_250630.csv'),
                'val_image_pair_path17' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_240930_final_rgbv2_update_250630.csv'),

                'val_image_pair_path18' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepinpad_241018_final_rgb_update_250630.csv'),
                'val_image_pair_path19' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_241018_final_rgb_update_250630.csv'),
                
                'val_image_pair_path20' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_241111_final_rgb_D433_update_250630.csv'),
                'val_image_pair_path21' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_241114_final_rgb_DA472_update_250630.csv'),
                'val_image_pair_path22' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_241206_final_rgb_DA534_update_250630.csv'),
                'val_image_pair_path23' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_241206_final_rgb_DA519_update_250630.csv'),
                'val_image_pair_path24' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_241220_final_rgb_DA1620_dropoldpairs_update_250630.csv'),
                'val_image_pair_path25' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_241231_final_rgb_DA3031_dropoldpairs_up_250522.csv'),

                # 典烨误报数据。仅用于测试
                'val_image_pair_path26' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250109_final_rgb_DA627-629_dropoldpairs_ori_up_250522.csv'),
                'val_image_pair_path27' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_train_pair_labels_singlepinpad_250109_final_rgb_DA627-629_dropoldpairs_ori_up_250522.csv')        ,

                # 辉瑞达特供数据, test_on_train
                # 开关漏报
                'val_image_pair_hrd_switch_clean4_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs_clean4.csv'),
                'val_image_pair_hrd_switch_clean4_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_train_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs_clean4.csv'),
                'val_image_pair_hrd_switch_clean4_path3': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_train_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs_clean4_ng_cross_pair_up.csv'),
                'val_image_pair_hrd_switch_clean4_path4': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs_clean4_ng_cross_pair_up.csv'),
                # 翘脚漏报
                'val_image_pair_hrd_led_qj_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj2_ng_dropoldpairs.csv'),
                'val_image_pair_hrd_led_qj_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj2_ng_dropoldpairs_ng_cross_pair.csv'),
                'val_image_pair_hrd_led_qj_path3': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj2_ok_dropoldpairs_ng_cross_pair.csv'),
                # 开关误报
                'val_image_pair_hrd_switch_fp_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250619_final_rgb_DA1232_hrd_switch_fp_dropoldpairs_ok_cross_pair_up.csv'),
                'val_image_pair_hrd_switch_fp_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250619_final_rgb_DA1232_hrd_switch_fp_dropoldpairs.csv'),
                                                   
                'val_image_pair_hrd_switch_fp_path3': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_train_pair_labels_singlepinpad_250622_final_rgb_DA1245_hrd_swith_fp_dropoldpairs.csv'),
                'val_image_pair_hrd_switch_fp_path4': os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250622_final_rgb_DA1245_hrd_swith_fp_dropoldpairs.csv'),
                                                   
                                                  

                'val_image_pair_hrd_test_on_train_path1' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'),
                
                'val_image_pair_hrd_test_on_train_path2' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv')  ,
                
                'val_image_pair_hrd_test_on_train_path3' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv') ,
                'val_image_pair_hrd_test_on_train_path4' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv') ,
                'val_image_pair_hrd_test_on_train_path5' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv') ,
                'val_image_pair_hrd_test_on_train_path6' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv')                           ,
                
                'val_image_pair_hrd_test_on_train_FAE_path1' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv'),
                # black_test
                'val_image_pair_hrd_black_test_path1' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250220_final_rgb_DA730_supply_dropoldpairs.csv')   ,
                'val_image_pair_hrd_black_test_path2' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_train_pair_labels_singlepinpad_250220_final_rgb_DA730_supply_dropoldpairs.csv') ,
                'val_image_pair_hrd_black_test_path3' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250222_final_rgb_DA743_dropoldpairs.csv')   ,
                'val_image_pair_hrd_black_test_path4' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_train_pair_labels_singlepinpad_250222_final_rgb_DA743_dropoldpairs.csv') ,
                'val_image_pair_hrd_black_test_path5' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_black_test_pair_labels_singlepinpad_250219_final_rgb_DA730_45_dropoldpairs_cross_pair.csv'),

                # 6,7是DA747原始数据未做筛选的全部测试
                'val_image_pair_hrd_black_test_path6' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_just_test.csv'),

                'val_image_pair_hrd_black_test_path7' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_just_test.csv'),
                
                # 6,7是DA747筛选后做cross pair的test on train结果
                'val_image_pair_hrd_black_test_path8' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),

                'val_image_pair_hrd_black_test_path9' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),
                # 6,7是DA747筛选后black test结果
                'val_image_pair_hrd_black_test_path10' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv'),
                

                # 辉瑞大现场2个漏报数据，临时测试
                'val_image_pair_path_hrd_1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_250222_final_rgb_DA743_dropoldpairs.csv')   ,
                # 辉瑞大现场2个漏报数据，临时测试
                'val_image_pair_path_hrd_ks_1' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs_clean3.csv') ,
        },
        'singlepad':{  
                # 广州华创冷焊
                'gzhc_lh1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs.csv'),
                'gzhc_lh2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs_ok_corss_pair_up.csv'), 
                
            
                #深圳迅航(TG)
                'szxh1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250728_final_rgb_DA1486_szxh_xh_op_dropoldpairs.csv'),
                'szxh2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250728_final_rgb_DA1486_szxh_xh_op_dropoldpairs.csv'), 
                'szxh3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250728_final_rgb_DA1486_szxh_xh_op_dropoldpairs_ng_corss_pair.csv'), 
                'szxh4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250728_final_rgb_DA1486_szxh_xh_op_dropoldpairs_ok_corss_pair_up.csv'), 
                #深圳朗特FAE(TG)
                'szlt_FAE1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_labels_singlepad_rgb_pairs_split_test_ok_cross_pair_up.csv'),
                'szlt_FAE2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_labels_singlepad_rgb_pairs_split_test_ng_cross_pair_up.csv'),    
                #深圳朗特clean                                                                                    
                'szlt_clean1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_labels_singlepad_rgb_pairs_split_train_ok_cross_pair_up_for_test.csv'),
                'szlt_clean2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_labels_singlepad_rgb_pairs_split_test_ok_cross_pair_up_clean.csv'),    
                'szlt_clean3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_labels_singlepad_rgb_pairs_split_test_ng_cross_pair_up_clean.csv'),   
                #深圳朗特TG                                                                                   
                'szlt_TG1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'defect_labels_singlepad_rgb_pairs_split_test_ok_cross_pair_up_TG.csv'),


                # 重庆国讯暗色漏报，特供
                'cqgx_data1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250701_final_rgb_DA1306_cqgx_dropoldpairs.csv'),
                'cqgx_data2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250701_final_rgb_DA1306_cqgx_dropoldpairs.csv'),                  
                              
                # 深圳硬姐虚焊少锡漏报
                'szyj_250702_data1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs.csv'),
                'szyj_250702_data2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs_ok_cross_pair_clean_up.csv'),                  
                'szyj_250702_data3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs_TG.csv'),
                'szyj_250702_data4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs_TG.csv'),


                # 南京菲林&泰克尔曼ok，该数据加入后模型效果降低，暂不加入一般数据集
                'njfl_tkr_clean1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_clean.csv'),
                'njfl_tkr_clean2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_ok_cross_pair_clean.csv'),                  
                'njfl_tkr_clean3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_clean.csv'),
                'njfl_tkr_clean4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_ok_cross_pair_clean.csv'),

                # 南京菲林&泰克尔曼特供                  
                'njfl_tkr_TG1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_TG.csv'),
                'njfl_tkr_TG2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_TG_ok_cross_pair_up.csv'),                  
                'njfl_tkr_TG3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_TG.csv'),
                'njfl_tkr_TG4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_TG_ok_cross_pair_up.csv'),                  
                                                
                # 惠州光弘
                'hzgh1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs.csv'),
                'hzgh2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs.csv'),  
                'hzgh3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_soft_TG_ok_cross_pair_up.csv'),
                'hzgh4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_hard_TG_ok_cross_pair_up.csv'),                                              
                
                # 炉前炉后
                'bp_stove_clean1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs.csv'),
                'bp_stove_clean2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_ok_cross_pair_up.csv'),
                'bp_stove_soft1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft.csv'),
                'bp_stove_soft2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft_ok_cross_pair.csv'),
                'bp_stove_TG1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_TG.csv'),
                'bp_stove_TG2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_TG_ok_cross_pair_up.csv'),
                                                 
                # 永林冷焊数据
                'yl_cold_solder1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs.csv'),
                'yl_cold_solder2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_cross_ng_diff_material_up.csv'), 
                'yl_cold_solder3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_ok_cross_pair_up.csv'),

                'yl_cold_solder4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs.csv'),

                'yl_cold_solder5': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up.csv'),

                'yl_cold_solder6': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_cross_ng_diff_material2_up.csv'),

                'yl_cold_solder7': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv'),

                'yl_cold_solder8': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv'),
                'yl_cold_solder9': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_ng_cross_pair.csv'),
                'yl_cold_solder_TG1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_TG.csv'),                                                    
                # 3D数据
                '3D_rgb1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250423_final_rgb_DA920_3D_dropoldpairs_clean.csv'),  
                # 金赛点软板错位
                'jsd_soft1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250507_final_rgb_DA978_dropoldpairs.csv'),
                'jsd_soft2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250507_final_rgb_DA978_dropoldpairs.csv'), 
                                                                                   
                # 全量原始数据筛选前uncertain数据
                'singlepad_before_pos_uncertain_samlpe_test': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'before_pos_uncertain_sample_results.csv'),
                # 全量黑盒数据，也包括训练测试中的uncertain
                'singlepad_all_black_sample_test': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'all_black_sample_results.csv'),
                # 全量原始数据筛选前uncertain数据
                'singlepad_before_pos_uncertain_test': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'all_singlepad_rgb_uncertain_before_pos_black_test_update.csv'),
                # 全量黑盒数据，也包括训练测试中的uncertain
                'singlepad_all_black_test': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'singlepad_all_black_test_update.csv'),
                # 深圳南方虚焊分数过低
                'val_image_pair_nf_op_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250327_final_rgb_DA825_op_dropoldpairs.csv'),
                'val_image_pair_nf_op_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250327_final_rgb_DA825_op_dropoldpairs.csv'),
                # 辉瑞大偏移漏报
                'val_image_pair_hrd_op_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250324_final_rgb_DA816_op_dropoldpairs_up_250702.csv'),
                'val_image_pair_jsd_fp_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250317_final_rgb_DA780_jsd_fp_dropoldpairs.csv'),
                'val_image_pair_jsd_fp_path2': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250317_final_rgb_DA780_jsd_fp_dropoldpairs.csv'),
                'val_image_pair_jsd_fp_path3': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs.csv'),
                'val_image_pair_jsd_fp_path4': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs_TG.csv'),

                'val_image_pair_path1': os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_val_pair_labels_singlepad_merged_up_250702.csv'),
                'val_image_pair_path2' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'bbtest_labels_pairs_singlepad_230306v2_up_250702.csv'),
                'val_image_pair_path3' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_val_pair_labels_singlepad_240329_final_update_250516.csv'),
                'val_image_pair_path4' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_240329_final_update_250516.csv'),
                'val_image_pair_path5' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'jiraissues_pair_labels_singlepad_240314_final_model_cleaned2_update_250512.csv'),
                'val_image_pair_path6' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'test_pair_labels_singlepad_240403debug_final_up_250702.csv'),
                'val_image_pair_path7' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'test_pair_labels_singlepad_240404debug_final_up_250702.csv'),
                
                'val_image_pair_path8' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_240428_final_RGB_update_250516.csv'),
                    
                'val_image_pair_path9' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_240429_final_RGB_update_250516.csv'),
                'val_image_pair_path11' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'jira_test_pair_labels_singlepad_240429_model_cleaned2_update_250512.csv'),
                'val_image_pair_path12' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_240507_final_RGB_model_cleaned2_update_250512.csv'),
                'val_image_pair_path100' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'bbtestmz_pair_purepad_updateaugmented_mz_up_250702.csv'),
                # val_image_pair_path13 : os.path.join(image_folder, 'merged_annotation', date,
                #                                     f'bbtestmz_pair_purepad_input_update_model_cleaned2.csv')
                # val_image_pair_path14 : os.path.join(image_folder, 'merged_annotation', date,
                #                                     f'aug_test_pair_labels_singlepad_240716_final_RGB_model_cleaned2.csv')
                'val_image_pair_path15' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_val_pair_labels_singlepad_240808_final_RGB_model_cleaned2_update_250512.csv'),
                'val_image_pair_path16' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_241018_final_rgb_update_250516.csv'),
                'val_image_pair_path17' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_241018_final_rgb_model_cleaned2.csv'),
                # 3D
                'val_image_pair_path18' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepad_241103_final_rgb_update_250516.csv'),
                'val_image_pair_path19' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepad_241111_final_rgb_D433_update_250516.csv'),
                'val_image_pair_path20' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepad_241114_final_rgb_DA465_update_250516.csv'),
                'val_image_pair_path21' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepad_241114_final_rgb_DA472_update_250516.csv'),

                'val_image_pair_path22' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepad_241126_final_rgb_DA505_update_250516.csv'),
                'val_image_pair_path23' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepad_241126_final_rgb_DA507_update_250516.csv'),

                'val_image_pair_path24' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepad_241127_final_rgb_DA509_update_250516.csv'),
                'val_image_pair_path25' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepad_241206_final_rgb_DA534_update_250512.csv'),
                'val_image_pair_path26' : os.path.join(image_folder, 'merged_annotation', date,
                                                            f'aug_test_pair_labels_singlepad_241206_final_rgb_DA519_update_250512.csv'),
                #############
                # 处理后的典烨数据，可加入一般测试
                'val_image_pair_path31' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs_update_250512.csv'),
                'val_image_pair_path32' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250110_final_rgb_DA629-2_dropoldpairs_update_250512.csv')                        ,
                
                # DA703昆山启佳特供数据，仅用于cur_jiraissue
                'val_image_pair_path33' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs.csv'),
                'val_image_pair_path34' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv')  ,
                
                'val_image_pair_path35' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250213_final_rgb_DA712_dropoldpairs.csv'),
                'val_image_pair_path36' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250213_final_rgb_DA712_dropoldpairs.csv')   ,
                
                'val_image_pair_path37' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs.csv'),
                'val_image_pair_path38' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs.csv') ,

                # 昆山启佳通用数据
                'val_image_pair_path39' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_general_final_rgb_hrd_ks_dropoldpairs_clean2.csv') ,

                # 黑盒uuid测试集(训练集测试集都用于测试)
                'val_image_pair_path40' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_250310_final_rgb_DA758_black_uuid_dropoldpairs.csv') ,
                'val_image_pair_path41' : os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_250310_final_rgb_DA758_black_uuid_dropoldpairs.csv') ,
        },
        
    }
    if part_name == 'padgroup':
        
        if valdataset == 'ry':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_ry_syh_rgb_path1'], algo_test_csvs[part_name]['val_image_pair_ry_syh_rgb_path2'], 
                                            ]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_ry_syh_white_path1'], algo_test_csvs[part_name]['val_image_pair_ry_syh_white_path2'],]
        elif valdataset == 'jsd':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_jsd_rgb_path1'], algo_test_csvs[part_name]['val_image_pair_jsd_rgb_path2'], 
                                            ]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_jsd_white_path1'], algo_test_csvs[part_name]['val_image_pair_jsd_white_path2'],]
        elif valdataset == 'hnzzzw':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['hnzzzw_rgb1'], algo_test_csvs[part_name]['hnzzzw_rgb2'], 
                                            ]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['hnzzzw_white1'], algo_test_csvs[part_name]['hnzzzw_white2'],]
        elif valdataset == 'hnzzzw_TG':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['hnzzzw_rgb_TG1'], algo_test_csvs[part_name]['hnzzzw_rgb_TG2'], 
                                            ]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['hnzzzw_white_TG1'], algo_test_csvs[part_name]['hnzzzw_white_TG2'],]                
        
        
        elif valdataset == '3D':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['3D_rgb1']]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['3D_white1']]

        elif valdataset == 'njzh':
                if img_color == 'rgb':
                    val_image_pair_path_list = [
                                                algo_test_csvs[part_name]['njzh_rgbwhite1'],
                                                algo_test_csvs[part_name]['njzh_rgbwhite2'],
                                                algo_test_csvs[part_name]['njzh_rgbwhite3'],
                                                ]
                elif img_color == 'white':
                    val_image_pair_path_list = [
                                                algo_test_csvs[part_name]['njzh_rgbwhite1'],
                                                algo_test_csvs[part_name]['njzh_rgbwhite2'],
                                                algo_test_csvs[part_name]['njzh_rgbwhite3'],
                        ]

                    
        elif valdataset == '3D_supply':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['3D_rgb_supply1']]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['3D_white_supply1']]
        elif valdataset == 'body_box_select': # 原深圳诚而信误报
                if img_color == 'rgb':
                    val_image_pair_path_list = [
                                                algo_test_csvs[part_name]['szcex_rgb_clean_fp1'],
                                                algo_test_csvs[part_name]['szcex_rgb_clean_fp2'],
                                            ]
                elif img_color == 'white':
                    val_image_pair_path_list = [ 
                                                algo_test_csvs[part_name]['szcex_white_clean_fp1'],
                                                algo_test_csvs[part_name]['szcex_white_clean_fp2'],
                    ]

        elif valdataset == 'szcc_lx_op':
                if img_color == 'rgb':
                    val_image_pair_path_list = [
                                                algo_test_csvs[part_name]['szcc_lx_rgb_op1'],
                                                algo_test_csvs[part_name]['szcc_lx_rgb_op2'],
                                                algo_test_csvs[part_name]['szcc_lx_white_op1'],
                                                algo_test_csvs[part_name]['szcc_lx_white_op2'],
                                            ]
                elif img_color == 'white':
                    val_image_pair_path_list = [ 
                                                algo_test_csvs[part_name]['szcc_lx_rgb_op1'],
                                                algo_test_csvs[part_name]['szcc_lx_rgb_op2'],
                                                algo_test_csvs[part_name]['szcc_lx_white_op1'],
                                                algo_test_csvs[part_name]['szcc_lx_white_op2'],
                    ]

        elif valdataset == 'szpj':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['szpj']]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['szpj']]
        elif valdataset == 'yj_lx_op':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['yj_lx_rgb_op1'],algo_test_csvs[part_name]['yj_lx_rgb_op2'],
                                                algo_test_csvs[part_name]['yj_lx_rgb_op3'],algo_test_csvs[part_name]['yj_lx_rgb_op4'],
                    ]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['yj_lx_white_op1'],algo_test_csvs[part_name]['yj_lx_white_op2'],
                                                algo_test_csvs[part_name]['yj_lx_white_op3'],algo_test_csvs[part_name]['yj_lx_white_op4'],
                    ]                                 
        elif valdataset == 'jxlzy':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['jxlzy_rgb1'], algo_test_csvs[part_name]['jxlzy_rgb2']]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['jxlzy_white1'], algo_test_csvs[part_name]['jxlzy_white2']]

        elif valdataset == 'jxlzy_TG':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['jxlzy_TG_rgb1'], algo_test_csvs[part_name]['jxlzy_TG_rgb2']]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['jxlzy_TG_white1'], algo_test_csvs[part_name]['jxlzy_TG_white2']]
        elif valdataset == 'zzzl':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['zzzl_rgb1'], algo_test_csvs[part_name]['zzzl_rgb2'],
                                                algo_test_csvs[part_name]['zzzl_rgb3'], algo_test_csvs[part_name]['zzzl_rgb4']
                                                ]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['zzzl_white1'], algo_test_csvs[part_name]['zzzl_white2'],
                                                algo_test_csvs[part_name]['zzzl_white3'], algo_test_csvs[part_name]['zzzl_white4']
                                                ]
        elif valdataset == 'xr_fp':

            if img_color == 'rgb':
                val_image_pair_path_list = [algo_test_csvs[part_name]['xr_rgb1'], algo_test_csvs[part_name]['xr_rgb2']]
            elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['xr_white1'], algo_test_csvs[part_name]['xr_white2']]   
        elif valdataset == 'sgjlc_250422':
            if img_color == 'rgb':
                val_image_pair_path_list = [algo_test_csvs[part_name]['sgjlc_rgb_250422']]
            elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['sgjlc_white_250422']]  

        elif valdataset == 'cur_jiraissue':
            if img_color == 'rgb':
                val_image_pair_path_list = [algo_test_csvs[part_name]['hnzzzw_rgb1'], algo_test_csvs[part_name]['hnzzzw_rgb2'], 
                                            ]
            elif img_color == 'white':
                val_image_pair_path_list = [algo_test_csvs[part_name]['hnzzzw_white1'], algo_test_csvs[part_name]['hnzzzw_white2'],
                                            ]                    
                           
        elif valdataset == 'jiraissues':
            if img_color == 'rgb':
                val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path20_rgb'],     algo_test_csvs[part_name]['val_image_pair_path23_rgb'], algo_test_csvs[part_name]['val_image_pair_path26_rgb'],
                                            algo_test_csvs[part_name]['val_image_pair_path28_rgb'], algo_test_csvs[part_name]['val_image_pair_path30_rgb'], 
                                            algo_test_csvs[part_name]['val_image_pair_path32_rgb'], algo_test_csvs[part_name]['val_image_pair_path_rgb35'], 
                                            algo_test_csvs[part_name]['val_image_pair_path_rgb37'], algo_test_csvs[part_name]['val_image_pair_path_rgb38'],
                                            algo_test_csvs[part_name]['val_image_pair_path_ks_hrd_general_rgb1'], algo_test_csvs[part_name]['val_image_pair_path_ks_hrd_general_rgb2'], 
                                            algo_test_csvs[part_name]['val_image_pair_path_ks_hrd_general_rgb3'],algo_test_csvs[part_name]['val_image_pair_yl_rgb_path1'], 
                                            algo_test_csvs[part_name]['val_image_pair_yl_rgb_path2'],algo_test_csvs[part_name]['dz_rgb1'], algo_test_csvs[part_name]['dz_rgb2'],
                                            algo_test_csvs[part_name]['xr_rgb1'],algo_test_csvs[part_name]['jxlzy_rgb1'], algo_test_csvs[part_name]['jxlzy_rgb2'],algo_test_csvs[part_name]['yj_lx_rgb_op1'],algo_test_csvs[part_name]['yj_lx_rgb_op2'],
                                                algo_test_csvs[part_name]['yj_lx_rgb_op3'],algo_test_csvs[part_name]['yj_lx_rgb_op4'],algo_test_csvs[part_name]['zzzl_rgb1'], algo_test_csvs[part_name]['zzzl_rgb2'],
                                            algo_test_csvs[part_name]['zzzl_rgb3'], algo_test_csvs[part_name]['zzzl_rgb4']
                                            ] 
                # 33,34算进bbtest, 对应val_image_pair_path_white47，48也算进bbtest    
            elif img_color == 'white':
                val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path38_white'],       algo_test_csvs[part_name]['val_image_pair_path39_white'], algo_test_csvs[part_name]['val_image_pair_path40_white'],
                                            algo_test_csvs[part_name]['val_image_pair_path46_white'],       algo_test_csvs[part_name]['val_image_pair_path42_white'], algo_test_csvs[part_name]['val_image_pair_path_white49'], 
                                            algo_test_csvs[part_name]['val_image_pair_path_white51'], algo_test_csvs[part_name]['val_image_pair_path_white52'],
                                            algo_test_csvs[part_name]['val_image_pair_path_ks_hrd_general_white1'], algo_test_csvs[part_name]['val_image_pair_path_ks_hrd_general_white2'], 
                                            algo_test_csvs[part_name]['val_image_pair_path_ks_hrd_general_white3'], algo_test_csvs[part_name]['val_image_pair_yl_white_path1'], 
                                            algo_test_csvs[part_name]['val_image_pair_yl_white_path2'],algo_test_csvs[part_name]['val_image_pair_ycy_white_path1'], 
                                            algo_test_csvs[part_name]['val_image_pair_ycy_white_path2'],algo_test_csvs[part_name]['dz_white1'], algo_test_csvs[part_name]['dz_white2'],
                                            algo_test_csvs[part_name]['xr_white1'],algo_test_csvs[part_name]['jxlzy_white1'], algo_test_csvs[part_name]['jxlzy_white2'],algo_test_csvs[part_name]['yj_lx_white_op1'],algo_test_csvs[part_name]['yj_lx_white_op2'],
                                                algo_test_csvs[part_name]['yj_lx_white_op3'],algo_test_csvs[part_name]['yj_lx_white_op4'],algo_test_csvs[part_name]['zzzl_white1'], algo_test_csvs[part_name]['zzzl_white2'],
                                                algo_test_csvs[part_name]['zzzl_white3'], algo_test_csvs[part_name]['zzzl_white4']     
                                            ]
        elif valdataset == 'graphite':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path_rgb400']]
                if img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path_rgb400']]
        elif valdataset == 'bbtest':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path7_rgb'], algo_test_csvs[part_name]['val_image_pair_path_rgb33'], 
                                                algo_test_csvs[part_name]['val_image_pair_path_gaoce1_rgb'],algo_test_csvs[part_name]['val_image_pair_ycy_rgb_path1'], 
                                                algo_test_csvs[part_name]['val_image_pair_ycy_rgb_path2'], algo_test_csvs[part_name]['val_image_pair_ycy_debug_rgb_path2'], 
                                            ]
                elif img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path_white47'], algo_test_csvs[part_name]['val_image_pair_path_white48'], 
                                                algo_test_csvs[part_name]['val_image_pair_path_gaoce3_white']]
        elif valdataset == 'debug':
                val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path26_rgb'], algo_test_csvs[part_name]['val_image_pair_path27_rgb']]
        elif valdataset == 'alltestval':
                if img_color == 'white':
                    val_image_pair_path_list = [ 
                        algo_test_csvs[part_name]['val_image_pair_path5_white'],  algo_test_csvs[part_name]['val_image_pair_path6_white'], algo_test_csvs[part_name]['val_image_pair_path33_white'],
                        algo_test_csvs[part_name]['val_image_pair_path34_white'], algo_test_csvs[part_name]['val_image_pair_path35_white'],algo_test_csvs[part_name]['val_image_pair_path36_white'],
                        algo_test_csvs[part_name]['val_image_pair_path37_white'], algo_test_csvs[part_name]['val_image_pair_path38_white'],algo_test_csvs[part_name]['val_image_pair_path39_white'],
                        algo_test_csvs[part_name]['val_image_pair_path40_white'], algo_test_csvs[part_name]['val_image_pair_path42_white'],algo_test_csvs[part_name]['val_image_pair_path43_white'],
                        algo_test_csvs[part_name]['val_image_pair_path44_white'], algo_test_csvs[part_name]['val_image_pair_path45_white'],algo_test_csvs[part_name]['val_image_pair_path46_white'],
                        algo_test_csvs[part_name]['val_image_pair_path_white47'], algo_test_csvs[part_name]['val_image_pair_path_white48']
                        ]
                elif img_color == 'rgb':
                    val_image_pair_path_list = [
                        algo_test_csvs[part_name]['val_image_pair_path1_rgb'], algo_test_csvs[part_name]['val_image_pair_path2_rgb'], algo_test_csvs[part_name]['val_image_pair_path3_rgb'],
                        algo_test_csvs[part_name]['val_image_pair_path4_rgb'], algo_test_csvs[part_name]['val_image_pair_path7_rgb'], algo_test_csvs[part_name]['val_image_pair_path8_rgb'],
                        algo_test_csvs[part_name]['val_image_pair_path9_rgb'], algo_test_csvs[part_name]['val_image_pair_path10_rgb'],algo_test_csvs[part_name]['val_image_pair_path12_rgb'],
                        algo_test_csvs[part_name]['val_image_pair_path13_rgb'],algo_test_csvs[part_name]['val_image_pair_path14_rgb'],algo_test_csvs[part_name]['val_image_pair_path15_rgb'],
                        algo_test_csvs[part_name]['val_image_pair_path16_rgb'],algo_test_csvs[part_name]['val_image_pair_path17_rgb'],algo_test_csvs[part_name]['val_image_pair_path18_rgb'],
                        algo_test_csvs[part_name]['val_image_pair_path19_rgb'],algo_test_csvs[part_name]['val_image_pair_path20_rgb'],#val_image_pair_path21,
                        algo_test_csvs[part_name]['val_image_pair_path22_rgb'],algo_test_csvs[part_name]['val_image_pair_path23_rgb'],algo_test_csvs[part_name]['val_image_pair_path24_rgb'],
                        algo_test_csvs[part_name]['val_image_pair_path25_rgb'],algo_test_csvs[part_name]['val_image_pair_path26_rgb'],algo_test_csvs[part_name]['val_image_pair_path27_rgb'],
                        algo_test_csvs[part_name]['val_image_pair_path28_rgb'],algo_test_csvs[part_name]['val_image_pair_path30_rgb'],
                        algo_test_csvs[part_name]['val_image_pair_path32_rgb'],algo_test_csvs[part_name]['val_image_pair_path_rgb33'],
                        
                    ]
        elif valdataset == 'newval':
                if img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path33_white'],algo_test_csvs[part_name]['val_image_pair_path38_white'],algo_test_csvs[part_name]['val_image_pair_path39_white'],
                                                algo_test_csvs[part_name]['val_image_pair_path40_white'],algo_test_csvs[part_name]['val_image_pair_path42_white'],algo_test_csvs[part_name]['val_image_pair_path46_white']]
                elif img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path3_rgb'],  algo_test_csvs[part_name]['val_image_pair_path4_rgb'],  algo_test_csvs[part_name]['val_image_pair_path8_rgb'],
                                                algo_test_csvs[part_name]['val_image_pair_path9_rgb'],  algo_test_csvs[part_name]['val_image_pair_path10_rgb'], algo_test_csvs[part_name]['val_image_pair_path12_rgb'],
                                                algo_test_csvs[part_name]['val_image_pair_path13_rgb'], algo_test_csvs[part_name]['val_image_pair_path14_rgb'], algo_test_csvs[part_name]['val_image_pair_path24_rgb'],
                                                algo_test_csvs[part_name]['val_image_pair_path26_rgb']]

        elif valdataset == 'oldval':
                if img_color == 'white':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path5_white'],  algo_test_csvs[part_name]['val_image_pair_path6_white'],  algo_test_csvs[part_name]['val_image_pair_path34_white'],
                                                algo_test_csvs[part_name]['val_image_pair_path35_white'], algo_test_csvs[part_name]['val_image_pair_path36_white'], algo_test_csvs[part_name]['val_image_pair_path37_white'],
                                                algo_test_csvs[part_name]['val_image_pair_path43_white'], algo_test_csvs[part_name]['val_image_pair_path44_white'], algo_test_csvs[part_name]['val_image_pair_path45_white']]
                elif img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path1_rgb'], algo_test_csvs[part_name]['val_image_pair_path2_rgb']]
    
        elif valdataset == 'okval':
                if img_color == 'rgb':
                    val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path8_rgb'],  algo_test_csvs[part_name]['val_image_pair_path9_rgb'], 
                                                algo_test_csvs[part_name]['val_image_pair_path10_rgb'], algo_test_csvs[part_name]['val_image_pair_path11_rgb']]
        
        elif valdataset == 'masked':
            if img_color == 'rgb':
                val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path15_rgb'],     algo_test_csvs[part_name]['val_image_pair_path16_rgb'], algo_test_csvs[part_name]['val_image_pair_path22_rgb'],
                                            algo_test_csvs[part_name]['val_image_pair_path25_rgb'],     algo_test_csvs[part_name]['val_image_pair_path17_rgb'], algo_test_csvs[part_name]['val_image_pair_path18_rgb'],
                                            algo_test_csvs[part_name]['val_image_pair_path19_rgb'],     algo_test_csvs[part_name]['val_image_pair_path27_rgb'], algo_test_csvs[part_name]['val_image_pair_path_rgb34'], 
                                            algo_test_csvs[part_name]['val_image_pair_path_rgb36'],     algo_test_csvs[part_name]['val_image_pair_path_rgb39'], 
                                            algo_test_csvs[part_name]['val_image_pair_path_rgb40'],     algo_test_csvs[part_name]['val_image_pair_path29_rgb'], algo_test_csvs[part_name]['val_image_pair_path31_rgb'], 
                                            algo_test_csvs[part_name]['3D_mask_rgb1'],algo_test_csvs[part_name]['jxlzy_mask_rgb1'],
                                            algo_test_csvs[part_name]['zzzl_rgb_mask1'], algo_test_csvs[part_name]['zzzl_rgb_mask2'],
                                            algo_test_csvs[part_name]['yj_lx_rgb_mask_op1'],algo_test_csvs[part_name]['yj_lx_rgb_mask_op2'], algo_test_csvs[part_name]['3D_white_mask_supply1'],

                                                    ]
            elif img_color == 'white':
                    val_image_pair_path_list = [
                        algo_test_csvs[part_name]['val_image_pair_path_white48'], algo_test_csvs[part_name]['val_image_pair_path_white50'], 
                        algo_test_csvs[part_name]['val_image_pair_path_white53'], algo_test_csvs[part_name]['val_image_pair_path_white54'],
                        algo_test_csvs[part_name]['3D_mask_white1'], algo_test_csvs[part_name]['jxlzy_mask_white1'],
                        algo_test_csvs[part_name]['zzzl_white_mask1'], algo_test_csvs[part_name]['zzzl_white_mask2'],
                    algo_test_csvs[part_name]['yj_lx_white_mask_op1'],algo_test_csvs[part_name]['yj_lx_white_mask_op2'], algo_test_csvs[part_name]['3D_white_mask_supply1'],


                        
                    ]
        elif valdataset == 'unacceptable_error':
            if img_color == 'rgb':
                val_image_pair_path_list = [algo_test_csvs[part_name]['extrem_rgb_ng'],algo_test_csvs[part_name]['jxlzy_rgb1'], algo_test_csvs[part_name]['jxlzy_rgb2']]
            elif img_color == 'white':
                val_image_pair_path_list = [algo_test_csvs[part_name]['extrem_white_ng'],algo_test_csvs[part_name]['jxlzy_white1'], algo_test_csvs[part_name]['jxlzy_white2']]
                
        elif valdataset == 'padgroup_test_rgb_all_data':
            val_image_pair_path_list = [x for x in list(algo_test_csvs[part_name].values()) if 'rgb' in x or 'RGB' in x]
        elif valdataset == 'padgroup_test_white_all_data':
            val_image_pair_path_list = [x for x in list(algo_test_csvs[part_name].values()) if 'white' in x or 'WHITE' in x]

    elif part_name == 'singlepinpad':
        if valdataset == 'bbtest':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path2']]
        elif valdataset == 'bbtestmz':
#             val_image_pair_path_list = [val_image_pair_path12, val_image_pair_path13]
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path100'], algo_test_csvs[part_name]['val_image_pair_hrd_op_path1']]

        elif valdataset == 'tjxt':
            val_image_pair_path_list = [algo_test_csvs[part_name]['tjxt_cold_solder1'], 
                                        algo_test_csvs[part_name]['tjxt_cold_solder2'],
                                        ]  
        elif valdataset == 'fzqzzl':
            val_image_pair_path_list = [algo_test_csvs[part_name]['fzqzzl_xh1'], 
                                        algo_test_csvs[part_name]['fzqzzl_xh2'],
                                        ]
            
        elif valdataset == 'szqswy_ori':
            val_image_pair_path_list = [algo_test_csvs[part_name]['szqswy_ori1'], 
                                        ]  

        elif valdataset == 'lh':
            val_image_pair_path_list = [algo_test_csvs[part_name]['gzhc_lh1'], 
                                        algo_test_csvs[part_name]['gzhc_lh2'],
                                        algo_test_csvs[part_name]['gzhc_lh3'], 
                                        algo_test_csvs[part_name]['gzhc_lh4'],
                                        algo_test_csvs[part_name]['tjxt_cold_solder1'], 
                                        algo_test_csvs[part_name]['tjxt_cold_solder2'],
                                        algo_test_csvs[part_name]['tjxt_cold_solder3'], 
                                        algo_test_csvs[part_name]['tjxt_cold_solder4'],
                                        algo_test_csvs[part_name]['tjxt_cold_solder5'], 
                                        algo_test_csvs[part_name]['tjxt_cold_solder6'],
                                        algo_test_csvs[part_name]['yl_cold_solder1'],
                                        algo_test_csvs[part_name]['yl_cold_solder2'], 
                                        algo_test_csvs[part_name]['yl_cold_solder3'], 
                                        algo_test_csvs[part_name]['yl_cold_solder4'],
                                        algo_test_csvs[part_name]['yl_cold_solder5'],
                                        algo_test_csvs[part_name]['yl_cold_solder6'], 
                                        algo_test_csvs[part_name]['yl_cold_solder7'], 
                                        algo_test_csvs[part_name]['yl_cold_solder8'],
                                        ]  
                                

        elif valdataset == 'szgkd':
            val_image_pair_path_list = [algo_test_csvs[part_name]['szgkd1'], 
                                        algo_test_csvs[part_name]['szgkd2'],
                                        ]
                                         
        elif valdataset == 'bp_stove_clean':
            val_image_pair_path_list = [algo_test_csvs[part_name]['bp_stove_clean1'], 
                                        algo_test_csvs[part_name]['bp_stove_clean2'],
                                        ]
      


        elif valdataset == '3D':
            val_image_pair_path_list = [algo_test_csvs[part_name]['3D_rgb1'], 
                                        ]

        
        elif valdataset == 'cur_jiraissue':
            val_image_pair_path_list = [ 
                                        algo_test_csvs[part_name]['mhdxh_250827_1'], 
                                        algo_test_csvs[part_name]['mhdxh_250827_2'],
                                        algo_test_csvs[part_name]['mhdxh_250827_3'], 
                                        algo_test_csvs[part_name]['mhdxh_250827_4'],
                                    # algo_test_csvs[part_name]['gzhc_lh1'], 
                                    # algo_test_csvs[part_name]['gzhc_lh2'],                  
                                        ]
        elif valdataset == 'mhdxh_250827_TG':
            val_image_pair_path_list = [ 
                                        algo_test_csvs[part_name]['mhdxh_250827_TG_1'], 
                                        algo_test_csvs[part_name]['mhdxh_250827_TG_2'],
                                    # algo_test_csvs[part_name]['gzhc_lh1'], 
                                    # algo_test_csvs[part_name]['gzhc_lh2'],                  
                                        ]

        elif valdataset == 'jiraissues':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path5'],  algo_test_csvs[part_name]['val_image_pair_path11'], algo_test_csvs[part_name]['val_image_pair_path19'],
                                        algo_test_csvs[part_name]['val_image_pair_path20'], algo_test_csvs[part_name]['val_image_pair_path21'], algo_test_csvs[part_name]['val_image_pair_path22'],
                                        algo_test_csvs[part_name]['val_image_pair_path23'], algo_test_csvs[part_name]['val_image_pair_path24'], algo_test_csvs[part_name]['val_image_pair_path25'],
                                        algo_test_csvs[part_name]['val_image_pair_path26'], algo_test_csvs[part_name]['val_image_pair_path27'], algo_test_csvs[part_name]['val_image_pair_rs_op_path1'],
                                        algo_test_csvs[part_name]['val_image_pair_rs_op_path2'],algo_test_csvs[part_name]['sgjlc1'], algo_test_csvs[part_name]['sgjlc2'],
                                        algo_test_csvs[part_name]['gznh1'], algo_test_csvs[part_name]['gznh2'], algo_test_csvs[part_name]['gznh2'],algo_test_csvs[part_name]['yl_cold_solder1'],
                                        algo_test_csvs[part_name]['yl_cold_solder2'], algo_test_csvs[part_name]['tjxt_cold_solder1'], algo_test_csvs[part_name]['tjxt_cold_solder2'],
                                        algo_test_csvs[part_name]['yl_cold_solder3'], algo_test_csvs[part_name]['yl_cold_solder4'], algo_test_csvs[part_name]['szfl_qj4'],
                                        algo_test_csvs[part_name]['szfl_qj1'], algo_test_csvs[part_name]['szfl_qj2'], algo_test_csvs[part_name]['szfl_qj3'],
                                        algo_test_csvs[part_name]['hzgh1'], algo_test_csvs[part_name]['jsks_xh_op2'],
                                        algo_test_csvs[part_name]['hzgh2'],
                                        algo_test_csvs[part_name]['ly_qj_250620_op1'], 
                                        algo_test_csvs[part_name]['ly_qj_250620_op2'],algo_test_csvs[part_name]['jsks_xh_op1'], 
                                        algo_test_csvs[part_name]['szgkd1'], 
                                        algo_test_csvs[part_name]['szgkd2'], algo_test_csvs[part_name]['szmhd_xh_op1'], 
                                        algo_test_csvs[part_name]['szmhd_xh_op2'],algo_test_csvs[part_name]['njml_clean1'], 
                                        algo_test_csvs[part_name]['njml_clean2'],algo_test_csvs[part_name]['njzh1'], 
                                        algo_test_csvs[part_name]['njzh2'],
                                        
                                        
                                        ]

        elif valdataset == 'alltestval':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path1'], algo_test_csvs[part_name]['val_image_pair_path3'],
                                        algo_test_csvs[part_name]['val_image_pair_path4'], algo_test_csvs[part_name]['val_image_pair_path5'],#val_image_pair_path6,
                                        algo_test_csvs[part_name]['val_image_pair_path7'], algo_test_csvs[part_name]['val_image_pair_path8'], algo_test_csvs[part_name]['val_image_pair_path9'],
                                        algo_test_csvs[part_name]['val_image_pair_path10'],algo_test_csvs[part_name]['val_image_pair_path11'],algo_test_csvs[part_name]['val_image_pair_path12'],
                                        algo_test_csvs[part_name]['val_image_pair_path14'],algo_test_csvs[part_name]['val_image_pair_path15'],algo_test_csvs[part_name]['val_image_pair_path16'],
                                        algo_test_csvs[part_name]['val_image_pair_path17'],algo_test_csvs[part_name]['val_image_pair_path18'],algo_test_csvs[part_name]['val_image_pair_path19'],
                                        algo_test_csvs[part_name]['val_image_pair_path20'],algo_test_csvs[part_name]['val_image_pair_path21'],algo_test_csvs[part_name]['val_image_pair_path22'],
                                        algo_test_csvs[part_name]['val_image_pair_path23'],algo_test_csvs[part_name]['val_image_pair_path24'],
                                        ]
        elif valdataset == 'newval':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path3'],  algo_test_csvs[part_name]['val_image_pair_path4'], 
                                        algo_test_csvs[part_name]['val_image_pair_path8'],  algo_test_csvs[part_name]['val_image_pair_path9'], 
                                        algo_test_csvs[part_name]['val_image_pair_path10'], algo_test_csvs[part_name]['val_image_pair_path19']]

        elif valdataset == 'gaoce':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path14'], algo_test_csvs[part_name]['val_image_pair_path15'], 
                                        algo_test_csvs[part_name]['val_image_pair_path16'], algo_test_csvs[part_name]['val_image_pair_path17']]
        elif valdataset == '3D':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path18'], algo_test_csvs[part_name]['val_image_pair_path19']]
        elif valdataset == 'debug':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path6']]
    
    elif part_name == 'singlepad':
        
        if valdataset == 'bbtest':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path2'],]
                                        
        elif valdataset == 'hzgh':
            val_image_pair_path_list = [algo_test_csvs[part_name]['hzgh1'], 
                                        algo_test_csvs[part_name]['hzgh2'],
                                        algo_test_csvs[part_name]['hzgh3'],

                                        ]
        elif valdataset == 'gzhc':
            val_image_pair_path_list = [algo_test_csvs[part_name]['gzhc_lh1'], 
                                        algo_test_csvs[part_name]['gzhc_lh2'],

                                        ]

        elif valdataset == 'lh':
            val_image_pair_path_list = [algo_test_csvs[part_name]['yl_cold_solder1'], 
                                        algo_test_csvs[part_name]['yl_cold_solder2'],
                                        algo_test_csvs[part_name]['yl_cold_solder3'],
                                        algo_test_csvs[part_name]['yl_cold_solder4'],
                                        algo_test_csvs[part_name]['yl_cold_solder5'],
                                        algo_test_csvs[part_name]['yl_cold_solder6'],
                                        algo_test_csvs[part_name]['yl_cold_solder7'], 
                                        algo_test_csvs[part_name]['yl_cold_solder8'],
                                        algo_test_csvs[part_name]['yl_cold_solder9'],
                                        algo_test_csvs[part_name]['gzhc_lh1'], 
                                        algo_test_csvs[part_name]['gzhc_lh2'],
                                        ]
    
        elif valdataset == 'szxh':
            val_image_pair_path_list = [algo_test_csvs[part_name]['szxh1'], 
                                        algo_test_csvs[part_name]['szxh2'],
                                        algo_test_csvs[part_name]['szxh3'],
                                        algo_test_csvs[part_name]['szxh4'],
                                        ]
            
        elif valdataset == 'cur_jiraissue':
            val_image_pair_path_list = [algo_test_csvs[part_name]['gzhc_lh1'], 
                                        algo_test_csvs[part_name]['gzhc_lh2'],
                                        ]
        elif valdataset == 'szlt_FAE':
            val_image_pair_path_list = [algo_test_csvs[part_name]['szlt_FAE1'], 
                                        algo_test_csvs[part_name]['szlt_FAE2'],

                                        ]
        elif valdataset == 'szlt_clean':
            val_image_pair_path_list = [algo_test_csvs[part_name]['szlt_clean1'], 
                                        algo_test_csvs[part_name]['szlt_clean2'],
                                        algo_test_csvs[part_name]['szlt_clean3'],


                                        ]
        elif valdataset == 'szlt_TG':
            val_image_pair_path_list = [algo_test_csvs[part_name]['szlt_TG1'], 

                                        ]
                                        
        elif valdataset == 'cqgx':
            val_image_pair_path_list = [algo_test_csvs[part_name]['cqgx_data1'], 
                                        algo_test_csvs[part_name]['cqgx_data2'],

                                        ]
                
        elif valdataset == 'szyj_250702':
            val_image_pair_path_list = [algo_test_csvs[part_name]['szyj_250702_data1'], 
                                        algo_test_csvs[part_name]['szyj_250702_data2'],
                                        ]    
        elif valdataset == 'szyj_250702_TG':
            val_image_pair_path_list = [algo_test_csvs[part_name]['szyj_250702_data3'], 
                                        algo_test_csvs[part_name]['szyj_250702_data4'],
                                        ]  
    
        elif valdataset == 'njfl_tkr_clean':
            val_image_pair_path_list = [algo_test_csvs[part_name]['njfl_tkr_clean1'], 
                                        algo_test_csvs[part_name]['njfl_tkr_clean2'],
                                        algo_test_csvs[part_name]['njfl_tkr_clean3'], 
                                        algo_test_csvs[part_name]['njfl_tkr_clean4'],
                                        ]
        elif valdataset == 'njfl_tkr_TG':
            val_image_pair_path_list = [algo_test_csvs[part_name]['njfl_tkr_TG1'], 
                                        algo_test_csvs[part_name]['njfl_tkr_TG2'],
                                        algo_test_csvs[part_name]['njfl_tkr_TG3'], 
                                        algo_test_csvs[part_name]['njfl_tkr_TG4'],
                                        ]
                        
        elif valdataset == 'hzgh_TG':
            val_image_pair_path_list = [algo_test_csvs[part_name]['hzgh4'], 

                                        ]
        elif valdataset == 'jsd_soft':
            val_image_pair_path_list = [algo_test_csvs[part_name]['jsd_soft1'], 
                                        algo_test_csvs[part_name]['jsd_soft2']]
        elif valdataset == 'bp_stove_clean':
            val_image_pair_path_list = [algo_test_csvs[part_name]['bp_stove_clean1'], 
                                        algo_test_csvs[part_name]['bp_stove_clean2']]            
        elif valdataset == 'bp_stove_soft':
            val_image_pair_path_list = [algo_test_csvs[part_name]['bp_stove_soft1'], 
                                        algo_test_csvs[part_name]['bp_stove_soft2']] 
        elif valdataset == 'bp_stove_TG':
            val_image_pair_path_list = [algo_test_csvs[part_name]['bp_stove_TG1'], 
                                        algo_test_csvs[part_name]['bp_stove_TG2']]             
        elif valdataset == 'yl_cold_solder':
            val_image_pair_path_list = [algo_test_csvs[part_name]['yl_cold_solder1'], 
                                        algo_test_csvs[part_name]['yl_cold_solder2'],
                                        algo_test_csvs[part_name]['yl_cold_solder3'],
                                        algo_test_csvs[part_name]['yl_cold_solder4'],
                                        algo_test_csvs[part_name]['yl_cold_solder5'],
                                        algo_test_csvs[part_name]['yl_cold_solder6'],
                                        algo_test_csvs[part_name]['yl_cold_solder7'], 
                                        algo_test_csvs[part_name]['yl_cold_solder8'],
                                        algo_test_csvs[part_name]['yl_cold_solder9'],
                                        ]

        elif valdataset == 'yl_cold_solder_TG':
            val_image_pair_path_list = [algo_test_csvs[part_name]['yl_cold_solder_TG1'],
                                        ]
        elif valdataset == 'calibrate_test':
            val_image_pair_path_list = [algo_test_csvs[part_name]['singlepad_all_black_sample_test'], 
                                        algo_test_csvs[part_name]['singlepad_before_pos_uncertain_samlpe_test']]
        elif valdataset == '3D':
            val_image_pair_path_list = [algo_test_csvs[part_name]['3D_rgb1']]            
            
        elif valdataset == 'before_pos_uncertain':
            val_image_pair_path_list = [algo_test_csvs[part_name]['singlepad_before_pos_uncertain_test']]
        elif valdataset == 'all_black':
            val_image_pair_path_list = [algo_test_csvs[part_name]['singlepad_all_black_test']]
        elif valdataset == 'hrd_op':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_hrd_op_path1']]
        elif valdataset == 'nf':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_nf_op_path1'], algo_test_csvs[part_name]['val_image_pair_nf_op_path2']]

        elif valdataset == 'jsd':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_jsd_fp_path1'], algo_test_csvs[part_name]['val_image_pair_jsd_fp_path2'],
                                        algo_test_csvs[part_name]['val_image_pair_jsd_fp_path3'], algo_test_csvs[part_name]['val_image_pair_jsd_fp_path4'],
                                        ]
            
        elif valdataset == 'black_uuid':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path40'], algo_test_csvs[part_name]['val_image_pair_path41']]

        elif valdataset == 'ksTG':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path33'], algo_test_csvs[part_name]['val_image_pair_path34'], 
                                        algo_test_csvs[part_name]['val_image_pair_path35'], algo_test_csvs[part_name]['val_image_pair_path36'], 
                                        algo_test_csvs[part_name]['val_image_pair_path37'], algo_test_csvs[part_name]['val_image_pair_path38']]

        elif valdataset == 'jiraissues':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path5'],  algo_test_csvs[part_name]['val_image_pair_path11'], algo_test_csvs[part_name]['val_image_pair_path12'], 
                                        algo_test_csvs[part_name]['val_image_pair_path15'], algo_test_csvs[part_name]['val_image_pair_path16'], algo_test_csvs[part_name]['val_image_pair_path18'], 
                                        algo_test_csvs[part_name]['val_image_pair_path19'], algo_test_csvs[part_name]['val_image_pair_path21'], algo_test_csvs[part_name]['val_image_pair_path20'],
                                        algo_test_csvs[part_name]['val_image_pair_path22'], algo_test_csvs[part_name]['val_image_pair_path23'], algo_test_csvs[part_name]['val_image_pair_path24'],
                                        algo_test_csvs[part_name]['val_image_pair_path25'], algo_test_csvs[part_name]['val_image_pair_path26'], algo_test_csvs[part_name]['val_image_pair_path31'], 
                                        algo_test_csvs[part_name]['val_image_pair_path32'],algo_test_csvs[part_name]['val_image_pair_path39'], algo_test_csvs[part_name]['szyj_250702_data1'], 
                                        algo_test_csvs[part_name]['szyj_250702_data2'],
                                        ]
        elif valdataset == 'bbtestmz':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path100'], algo_test_csvs[part_name]['val_image_pair_hrd_op_path1']]

        elif valdataset == 'alltestval':
            val_image_pair_path_list = [
                                        algo_test_csvs[part_name]['val_image_pair_path1'], 
                                        algo_test_csvs[part_name]['val_image_pair_path100'], algo_test_csvs[part_name]['val_image_pair_hrd_op_path1'],
                                        algo_test_csvs[part_name]['val_image_pair_path6'], algo_test_csvs[part_name]['val_image_pair_path7'],
                                        ]

        elif valdataset == 'newval':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path3'],  algo_test_csvs[part_name]['val_image_pair_path4'],  algo_test_csvs[part_name]['val_image_pair_path8'], 
                                        algo_test_csvs[part_name]['val_image_pair_path9'],  algo_test_csvs[part_name]['val_image_pair_path16'], algo_test_csvs[part_name]['val_image_pair_path18'], 
                                        algo_test_csvs[part_name]['val_image_pair_path19'], algo_test_csvs[part_name]['val_image_pair_path21'], algo_test_csvs[part_name]['val_image_pair_path20'],
                                        algo_test_csvs[part_name]['val_image_pair_path22'], algo_test_csvs[part_name]['val_image_pair_path23'], algo_test_csvs[part_name]['val_image_pair_path24']]

        elif valdataset == 'yijingdata':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path6'], algo_test_csvs[part_name]['val_image_pair_path7'], algo_test_csvs[part_name]['val_image_pair_path12']]

        elif valdataset == '3D_old':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path16'], algo_test_csvs[part_name]['val_image_pair_path17'], algo_test_csvs[part_name]['val_image_pair_path18'], 
                                        algo_test_csvs[part_name]['val_image_pair_path19'], algo_test_csvs[part_name]['val_image_pair_path21'], algo_test_csvs[part_name]['val_image_pair_path20'],
                                        algo_test_csvs[part_name]['val_image_pair_path22'], algo_test_csvs[part_name]['val_image_pair_path23'], algo_test_csvs[part_name]['val_image_pair_path24']]
        elif valdataset == 'tmptcheck':
            val_image_pair_path1000 = os.path.join(image_folder, 'merged_annotation', date,
                                            f'aug_debug_pair_labels_singlepad_241125.csv')
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path1000']]

    elif part_name == 'body':      
        if valdataset == 'bbtest':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path3']]
        elif valdataset == 'blacktestmz':
            val_image_pair_path_list = [algo_test_csvs[part_name]['blacktestmz']]
        elif valdataset == 'szlsd':
            val_image_pair_path_list = [algo_test_csvs[part_name]['szlsd_wrong1'],
                                        algo_test_csvs[part_name]['szlsd_wrong2'],
                                        ]
        elif valdataset == 'gzls':
            val_image_pair_path_list = [algo_test_csvs[part_name]['gzls_chip_1'],
                                        algo_test_csvs[part_name]['gzls_chip_2'],
                                        algo_test_csvs[part_name]['gzls_chip_3'],
                                        algo_test_csvs[part_name]['gzls_chip_4'],
                                        ]            

            
        elif valdataset == 'jsd_soft':
            val_image_pair_path_list = [algo_test_csvs[part_name]['jsd_soft1'], algo_test_csvs[part_name]['jsd_soft2']]            
        elif valdataset == 'gc_250520':
            val_image_pair_path_list = [algo_test_csvs[part_name]['gc_wrong1'], algo_test_csvs[part_name]['gc_wrong2']]     
        elif valdataset == 'qb':
            val_image_pair_path_list = []             
        elif valdataset == 'dwzts':
            val_image_pair_path_list = []   

            
        elif valdataset == 'body_CONCAT':
            val_image_pair_path_list = [algo_test_csvs[part_name]['body_CONCAT']]
        elif valdataset == 'body_OVER_SOLDER':
            val_image_pair_path_list = [algo_test_csvs[part_name]['body_OVER_SOLDER']]
        elif valdataset == 'body_SEHUAN_NG':
            val_image_pair_path_list = [algo_test_csvs[part_name]['body_SEHUAN_NG']]
        elif valdataset == 'body_SEHUAN_OK':
            val_image_pair_path_list = [algo_test_csvs[part_name]['body_SEHUAN_OK']]
        elif valdataset == 'body_WORD_PAIR':
            val_image_pair_path_list = [algo_test_csvs[part_name]['body_WORD_PAIR']]
                       
        elif valdataset == 'bbtestmz':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path100'], 
                                        algo_test_csvs[part_name]['val_image_pair_path20'], algo_test_csvs[part_name]['val_image_pair_path21'], algo_test_csvs[part_name]['val_image_pair_path24'], 
                                        algo_test_csvs[part_name]['val_image_pair_path25'], algo_test_csvs[part_name]['val_image_pair_path26'], algo_test_csvs[part_name]['val_image_pair_path27']]
            
        elif valdataset == 'cur_jiraissue':  #
            val_image_pair_path_list = [algo_test_csvs[part_name]['gzls_chip_1'],
                                        algo_test_csvs[part_name]['gzls_chip_2'],
                                        algo_test_csvs[part_name]['gzls_chip_3'],
                                        algo_test_csvs[part_name]['gzls_chip_4'],]

        elif valdataset == 'bdg':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path17'],
                                        algo_test_csvs[part_name]['val_image_pair_path17_1']
                                        ]
            
        elif valdataset == 'jiraissues':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path10'],  algo_test_csvs[part_name]['val_image_pair_path11'],  algo_test_csvs[part_name]['val_image_pair_path12'],
                                        algo_test_csvs[part_name]['val_image_pair_path13'],  algo_test_csvs[part_name]['val_image_pair_path14'],  algo_test_csvs[part_name]['val_image_pair_path19'], 
                                        algo_test_csvs[part_name]['val_image_pair_path22'],  algo_test_csvs[part_name]['val_image_pair_gaoce_1'], algo_test_csvs[part_name]['val_image_pair_gaoce_2'],
                                        algo_test_csvs[part_name]['val_image_pair_gaoce_3'], algo_test_csvs[part_name]['val_image_pair_gaoce_4'], algo_test_csvs[part_name]['val_image_pair_aug_ng_2'], 
                                        algo_test_csvs[part_name]['val_image_pair_aug_ng_4'],algo_test_csvs[part_name]['qb_wrong1'], algo_test_csvs[part_name]['qb_wrong2'],algo_test_csvs[part_name]['val_image_pair_jlc_path1'], 
                                        algo_test_csvs[part_name]['val_image_pair_jlc_path2'],algo_test_csvs[part_name]['val_image_pair_796_797_tg_path1'],algo_test_csvs[part_name]['val_image_pair_796_797_path1'],
                                        algo_test_csvs[part_name]['val_image_pair_796_797_path2'],algo_test_csvs[part_name]['val_image_pair_yj_path1'],algo_test_csvs[part_name]['val_image_pair_yj_path2'],
                                        algo_test_csvs[part_name]['dwzts_wrong12'], algo_test_csvs[part_name]['dwzts_wrong11'],
                                        algo_test_csvs[part_name]['dwzts_wrong13'],algo_test_csvs[part_name]['dwzts_wrong1'], algo_test_csvs[part_name]['dwzts_wrong2'],
                                        algo_test_csvs[part_name]['dwzts_wrong3'], algo_test_csvs[part_name]['dwzts_wrong4'],
                                        algo_test_csvs[part_name]['dwzts_wrong5'], algo_test_csvs[part_name]['dwzts_wrong6'],
                                        algo_test_csvs[part_name]['dwzts_wrong7'], algo_test_csvs[part_name]['dwzts_wrong8'],
                                        algo_test_csvs[part_name]['dwzts_wrong9'], algo_test_csvs[part_name]['dwzts_wrong10'],algo_test_csvs[part_name]['gc_wrong1'], algo_test_csvs[part_name]['gc_wrong2']
]
        elif valdataset == 'gaoce_led':
             val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path13'], algo_test_csvs[part_name]['val_image_pair_path14']]  

        elif valdataset == 'gaoce_trured_over':
             val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path20'], algo_test_csvs[part_name]['val_image_pair_path21'], algo_test_csvs[part_name]['val_image_pair_path24'],
                                         algo_test_csvs[part_name]['val_image_pair_path25'], algo_test_csvs[part_name]['val_image_pair_path26'], algo_test_csvs[part_name]['val_image_pair_path27']] 
             
        elif valdataset == 'alltestval':
            val_image_pair_path_list = [
                                        algo_test_csvs[part_name]['val_image_pair_path1'],  algo_test_csvs[part_name]['val_image_pair_path4'],  algo_test_csvs[part_name]['val_image_pair_path5'],
                                        algo_test_csvs[part_name]['val_image_pair_path7'],  algo_test_csvs[part_name]['val_image_pair_path8'],  algo_test_csvs[part_name]['val_image_pair_path9'],  
                                        algo_test_csvs[part_name]['val_image_pair_path10'], algo_test_csvs[part_name]['val_image_pair_path11'], algo_test_csvs[part_name]['val_image_pair_path12'],
                                        algo_test_csvs[part_name]['val_image_pair_path14'], algo_test_csvs[part_name]['val_image_pair_path13'], 
                                        algo_test_csvs[part_name]['val_image_pair_path15'], algo_test_csvs[part_name]['val_image_pair_path16'],
                                        ]
        elif valdataset == 'newval':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path4'], algo_test_csvs[part_name]['val_image_pair_path5'], algo_test_csvs[part_name]['val_image_pair_path7']]
        elif valdataset == 'debug':
            val_image_pair_path_list = [algo_test_csvs[part_name]['val_image_pair_path99']]   


        elif valdataset == 'body_train_nonpair2s':
            val_image_pair_path_list = [os.path.join(image_folder, 'merged_annotation', date,
                                                    f'aug_train_pair_labels_body_240913_final_white_model_cleaned_250306.csv') ]   
        elif 'body_all_data' in valdataset:
            val_image_pair_path_list = list(set(algo_test_csvs[part_name].values()))

            from train_data import get_train_csv
            image_folder = os.path.join(image_folder, 'merged_annotation', date)
            annotation_filename, val_annotation_filename, test_annotation_filename, aug_train_pair_data_filenames, aug_val_pair_data_filenames = \
                get_train_csv(image_folder, part_name, 'bodywhiteslimf1newonlybdg')
            
            val_image_pair_path_list += aug_train_pair_data_filenames
            val_image_pair_path_list += aug_val_pair_data_filenames

            val_image_pair_path_list += [annotation_filename, val_annotation_filename, test_annotation_filename]

            total_nums = len(val_image_pair_path_list)
            print(f'body_all_data nums: {total_nums}')
            val_image_pair_path_list = list(set(val_image_pair_path_list))
            import pickle
            body_all_data1_csvs = './body_all_data1.pkl'
            body_all_data2_csvs = './body_all_data2.pkl'
            body_all_data3_csvs = './body_all_data3.pkl'
            body_all_data4_csvs = './body_all_data4.pkl'
            body_all_data5_csvs = './body_all_data5.pkl'

            if valdataset == 'body_all_data1':
                val_image_pair_path_list = val_image_pair_path_list[:total_nums//3]
                with open(body_all_data1_csvs, 'wb') as f:
                    pickle.dump(val_image_pair_path_list, f)
            
            elif valdataset == 'body_all_data2':
                with open(body_all_data1_csvs, 'rb') as f:
                    body_all_data1_list = pickle.load(f)
                val_image_pair_path_list = [x for x in val_image_pair_path_list if x not in body_all_data1_list]
                body_all_data2_list_nums= len(val_image_pair_path_list)

                val_image_pair_path_list = val_image_pair_path_list[:body_all_data2_list_nums//3]

                body_all_data2_list = body_all_data1_list + val_image_pair_path_list
                with open(body_all_data2_csvs, 'wb') as f:
                    pickle.dump(body_all_data2_list, f)

            elif valdataset == 'body_all_data3':
                with open(body_all_data2_csvs, 'rb') as f:
                    body_all_data2_list = pickle.load(f)
                val_image_pair_path_list = [x for x in val_image_pair_path_list if x not in body_all_data2_list]
                body_all_data3_list_nums= len(val_image_pair_path_list)

                val_image_pair_path_list = val_image_pair_path_list[:body_all_data3_list_nums//2]

                body_all_data3_list = body_all_data2_list + val_image_pair_path_list
                with open(body_all_data3_csvs, 'wb') as f:
                    pickle.dump(body_all_data3_list, f)

            elif valdataset == 'body_all_data4':
                with open(body_all_data3_csvs, 'rb') as f:
                    body_all_data3_list = pickle.load(f)
                val_image_pair_path_list = [x for x in val_image_pair_path_list if x not in body_all_data3_list]
                body_all_data4_list_nums= len(val_image_pair_path_list)

                val_image_pair_path_list = val_image_pair_path_list[:body_all_data4_list_nums//2]

                body_all_data4_list = body_all_data3_list + val_image_pair_path_list
                with open(body_all_data4_csvs, 'wb') as f:
                    pickle.dump(body_all_data4_list, f)

            elif valdataset == 'body_all_data5':
                with open(body_all_data4_csvs, 'rb') as f:
                    body_all_data4_list = pickle.load(f)
                val_image_pair_path_list = [x for x in val_image_pair_path_list if x not in body_all_data4_list]
                body_all_data5_list_nums= len(val_image_pair_path_list)
                print(body_all_data5_list_nums)
                val_image_pair_path_list = val_image_pair_path_list[:3]

                body_all_data5_list = body_all_data4_list + val_image_pair_path_list
                with open(body_all_data5_csvs, 'wb') as f:
                    pickle.dump(body_all_data5_list, f)

            elif valdataset == 'body_all_data6':
                with open(body_all_data5_csvs, 'rb') as f:
                    body_all_data5_list = pickle.load(f)
                val_image_pair_path_list = [x for x in val_image_pair_path_list if x not in body_all_data5_list]

    return val_image_pair_path_list


