import os
import cv2
import subprocess

import multiprocessing
import subprocess

import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from glob import glob
from concurrent.futures import ProcessPoolExecutor

def visualize_pair(ref_img, insp_img, insp_label, binary_label, certainty, fig_path):
    ref_image_show = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    insp_image_show = cv2.cvtColor(insp_img, cv2.COLOR_BGR2RGB)

    # plot reference image against inspected image
    if ref_img.shape[1] / ref_img.shape[0] > 2.5:
        fig, axes = plt.subplots(2, 1, figsize=(4, 6))

    else:
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))


    axes[0].imshow(ref_image_show, cmap='gray')
    axes[0].set_title(f'Ref Binary NG = {binary_label} ({certainty})')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(insp_image_show, cmap='gray')
    axes[1].set_title(f'Insp Mclass = {insp_label}')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.savefig(fig_path)
    plt.close()

def vis_defect_pairs(img_folder,  date,  region, datasplit):
    csv_files = []
    if region == 'singlepad':
        csv_files += [
                                    # os.path.join(img_folder, 'merged_annotation', date,
                                    #             f'aug_train_pair_labels_{region}_merged.csv'),
                                    # os.path.join(img_folder, 'merged_annotation', date,
                                    #             f'aug_train_pair_labels_{region}_240329_final.csv'),
                                    # # os.path.join(img_folder,
                                    # #              f'aug_train_pair_labels_{region}_240403debug_final.csv'),
                                    # # os.path.join(img_folder,
                                    # #              f'aug_train_pair_labels_{region}_240404debug_final.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #             f'aug_train_pair_labels_{region}_240428_final_RGB.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #             f'aug_train_pair_labels_{region}_240429_final_RGB.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #             f'aug_train_pair_labels_{region}_240507_final_RGB.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #             f'aug_train_pair_labels_{region}_240808_final_RGB.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #             f'aug_train_pair_labels_{region}_241018_final_rgb.csv'),
                                    # 241111 241114
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #                 f'aug_train_pair_labels_singlepad_241103_final_rgb.csv'),
                                    os.path.join(img_folder,'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_241111_final_rgb_D433.csv'),
                                    os.path.join(img_folder,'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_241114_final_rgb_DA465.csv'),
                                    os.path.join(img_folder,'merged_annotation', date,
                                                    f'aug_train_pair_labels_singlepad_241114_final_rgb_DA472.csv'),
                                        ]
        csv_files += [
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #                 f'aug_val_pair_labels_{region}_merged.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date, 
                                    #                 f'aug_val_pair_labels_{region}_240329_final.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date, 
                                    #             f'aug_test_pair_labels_{region}_240329_final.csv'),
                                    # # os.path.join(img_folder,
                                    # #              f'aug_val_pair_labels_{region}_240403debug_final.csv'),
                                    # # os.path.join(img_folder,
                                    # #              f'aug_val_pair_labels_{region}_240404debug_final.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #             f'aug_test_pair_labels_{region}_240428_final_RGB.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #             f'aug_test_pair_labels_{region}_240429_final_RGB.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #             f'aug_val_pair_labels_{region}_240808_final_RGB.csv'),
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #             f'aug_test_pair_labels_{region}_241018_final_rgb.csv'),
                                    # # 3D lighting  
                                    # os.path.join(img_folder,'merged_annotation', date,
                                    #                 f'aug_test_pair_labels_singlepad_241103_final_rgb.csv'),
                                    os.path.join(img_folder,'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_241111_final_rgb_D433.csv'),
                                    os.path.join(img_folder,'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_241114_final_rgb_DA465.csv'),
                                    os.path.join(img_folder,'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepad_241114_final_rgb_DA472.csv')
                                    ]
    
        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_code_val = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}    
        
    elif region == 'singlepinpad':
        csv_files += [
                                        # os.path.join(img_folder,'merged_annotation', date,
                                        #              f'aug_train_pair_labels_{region}_merged.csv'),
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_{region}_240329_final.csv'),
                                        #  # os.path.join(img_folder,'merged_annotation', date,
                                        #  #              f'aug_train_pair_labels_{region}_240403debug_final.csv'),
                                        #  # os.path.join(img_folder,'merged_annotation', date,
                                        #  #              f'aug_train_pair_labels_{region}_240404debug_final.csv'),
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_{region}_240424_final_RGB.csv'),
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_{region}_240428_final_RGB.csv'),
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_{region}_240429_final_RGB.csv'),
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_singlepinpad_240715_final_RGB.csv'),
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_singlepinpad_240702_final_RGB.csv'),
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_singlepinpad_240708_final_RGB.csv'),
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_singlepinpad_240725_final_rgb.csv'),
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_singlepinpad_240930_final_rgbv2.csv'),
                                        #  # 3D lighting                                       
                                        #  os.path.join(img_folder,'merged_annotation', date,
                                        #               f'aug_train_pair_labels_singlepinpad_241018_final_rgb.csv'),

                                                      
                                         # 241111, 241114
                                         os.path.join(img_folder,'merged_annotation', date,
                                                      f'aug_train_pair_labels_singlepinpad_241111_final_rgb_D433.csv'),
                                         os.path.join(img_folder,'merged_annotation', date,
                                                      f'aug_train_pair_labels_singlepinpad_241114_final_rgb_DA472.csv'),
                                         os.path.join(img_folder,'merged_annotation', date,
                                                      f'aug_train_pair_labels_singlepinpad_241114_final_rgb_DA465.csv'),
                                         
                                         ]

       
        csv_files += [
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_{region}_240424_final_RGB_NG.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_{region}_240428_final_RGB_NG.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_{region}_240429_final_RGB_NG.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #          f'aug_train_pair_labels_singlepinpad_240715_final_RGBng.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #          f'aug_train_pair_labels_singlepinpad_240702_final_RGBng.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #          f'aug_train_pair_labels_singlepinpad_240708_final_RGBng.csv')
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_ng.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240913_final_rgb_ng.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240919_final_rgb_ng.csv'),
            ]

        csv_files += [
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_{region}_240424_final_RGBdownsampled.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_{region}_240428_final_RGBdownsampled.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_{region}_240429_final_RGBdownsampled.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240715_final_RGB.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240702_final_RGB.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240708_final_RGB.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240725_final_rgb.csv'),
            ]
        csv_files += [
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_{region}_240424_final_RGB.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_{region}_240428_final_RGB.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_{region}_240429_final_RGB.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240715_final_RGB.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240702_final_RGB.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240708_final_RGB.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240725_final_rgb.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240913_final_rgb.csv'),
                # os.path.join(img_folder,'merged_annotation', date,
                #              f'aug_train_pair_labels_singlepinpad_240919_final_rgb.csv'),
            ]

        csv_files += [
                                        # os.path.join(img_folder,'merged_annotation', date,
                                        #              f'aug_val_pair_labels_{region}_merged.csv'),
                                        # os.path.join(img_folder,'merged_annotation', date,
                                        #              f'aug_val_pair_labels_{region}_240329_final.csv'),
                                        # os.path.join(img_folder,'merged_annotation', date, f'aug_test_pair_labels_{region}_240329_final.csv'),
                                        # os.path.join(img_folder,'merged_annotation', date,
                                        #                 f'aug_test_pair_labels_{region}_240424_final_RGB.csv'),
                                        # os.path.join(img_folder,'merged_annotation', date,
                                        #                 f'aug_test_pair_labels_{region}_240428_final_RGB.csv'),
                                        # os.path.join(img_folder,'merged_annotation', date,
                                        #                 f'aug_test_pair_labels_{region}_240429_final_RGB.csv'),
                                        # os.path.join(img_folder,'merged_annotation', date,
                                        #                 f'aug_test_pair_labels_{region}_240716_final_RGB.csv'),
                                        # # 3D lighting                                 
                                        # os.path.join(img_folder,'merged_annotation', date,
                                        #                 f'aug_test_pair_labels_{region}_241018_final_rgb.csv'),
                                        os.path.join(img_folder,'merged_annotation', date,
                                                        f'aug_test_pair_labels_singlepinpad_241111_final_rgb_D433.csv'),
                                        os.path.join(img_folder,'merged_annotation', date,
                                                    f'aug_test_pair_labels_singlepinpad_241114_final_rgb_DA472.csv')
                                       ]
        
        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_code_val = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}

    defect_code_revert = {v: k for k, v in defect_code.items()}

    defect_code_link = {v: i for i, v in enumerate(defect_code.values())}
    defect_class_considered = list(defect_code_link.keys())

    wrong_csv = []
    for csv_file_path in csv_files:
        print(f"processing {csv_file_path}")
        csv_name = os.path.basename(csv_file_path)
        csv_name_ = csv_name.split('.')[0]
        datasplit = csv_name.split('_')[1]


        csv_df = pd.read_csv(csv_file_path)

        for index, csv_info in csv_df.iterrows():
            if csv_info['insp_y_raw'] not in defect_class_considered:
                continue
            insp_y_raw = csv_info['insp_y_raw']
            csv_insp_y = csv_info['insp_y']
            if csv_insp_y == -1:
                continue
            raw_defect = defect_code_revert[insp_y_raw]

            defect_code_link_insp_y = defect_code_link[insp_y_raw]

            if defect_code_link_insp_y:
                if csv_name not in wrong_csv:
                    wrong_csv.append(wrong_csv)
                wrong_csv_path = os.path.join(f'./defect_pairs_241111/{region}/{csv_name_}/{raw_defect}')
                datasplit_vis = os.path.join(f'./defect_pairs_241111/{region}/{datasplit}/{raw_defect}')

                os.makedirs(wrong_csv_path, exist_ok=True)
                os.makedirs(datasplit_vis, exist_ok=True)

                ref_img_relative_path = csv_info['ref_image']
                insp_img_relative_path = csv_info['insp_image']

                ref_img_path = os.path.join(img_folder, ref_img_relative_path)
                insp_img_path = os.path.join(img_folder, insp_img_relative_path)

                ref_img = cv2.imread(ref_img_path)
                insp_img = cv2.imread(insp_img_path)
                
                if 'ref_uuid' not in csv_info.keys():
                    ref_uuid = csv_info['ref_image_name'].replace('.png', '')
                    insp_uuid = csv_info['insp_image_name'].replace('.png', '')
                    if len(ref_uuid) > 40:
                        ref_uuid = index
                        insp_uuid = insp_uuid
                else:
                    ref_uuid = csv_info['ref_uuid']
                    insp_uuid = csv_info['insp_uuid']

                insp_xy = csv_info['insp_xy']
                ref_xy = csv_info['ref_xy']

                defect_label = csv_info['insp_defect_label']
                         # f'{issue_string}-{insp_uuid}_xy{insp_xy}_vs_{ref_uuid}_xy{ref_xy}.jpg'
                filename = f"{defect_label}_{insp_uuid}_xy{insp_xy}_vs_{ref_uuid}_xy{ref_xy}.png"
                fig_path_contain_csv = os.path.join(wrong_csv_path, filename)
                fig_path = os.path.join(datasplit_vis, filename)

                certainy = csv_info['confidence']
                visualize_pair(ref_img, insp_img, defect_label, raw_defect, certainy, fig_path_contain_csv)
                visualize_pair(ref_img, insp_img, defect_label, raw_defect, certainy, fig_path)

                # ref_img, insp_img, insp_label, binary_label, certainty, fig_path

def show_changed_singlepad(csv_path, data_path, save_pair_path):
    os.makedirs(save_pair_path, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    for index, infos in df.iterrows():
        if index == 130:
            print('sad')
        ref_img, insp_img = infos['ref_image'], infos['insp_image']

        ref_img = cv2.imread(os.path.join(data_path, ref_img))
        insp_img = cv2.imread(os.path.join(data_path, insp_img))

        old_defect = infos['old_defect']
        new_defect = infos['new_defect']
        
        old_confidence = infos['old_confidence']
        new_confidence = infos['new_confidence']

        certain = f"{old_confidence}_vs_{new_confidence} \n {old_defect}_vs_{new_defect}"
        fig_name = f"{index}_{old_defect}_vs_{new_defect}.png"
        sub_dif = os.path.join(save_pair_path, f'{old_confidence}_vs_{new_confidence}_{old_defect}_vs_{new_defect}')
        os.makedirs(sub_dif, exist_ok=True)
        fig_path = os.path.join(sub_dif, fig_name)
        visualize_pair(ref_img, insp_img, new_defect, old_defect, certain, fig_path)



def gather_model_predict_wrong(csv_dir_path, data_path, cur_color, target_color):
    predict_wrong_lists = []
    for csv_path in glob(os.path.join(csv_dir_path, "*", "*_ng.csv")):
        cur_df = pd.read_csv(csv_path)
        datasets = csv_path.split('/')[-2]

        model_predict_wrong_df = cur_df[~cur_df['model_predict_right']]
        if model_predict_wrong_df.empty:
            continue
        if 'insp_defect_label' in model_predict_wrong_df.columns:
            defect_type = 'insp_defect_label'
        else:
            pass
        info = model_predict_wrong_df[['ref_image', 'insp_image', defect_type, 'binary_y', 'insp_y_raw', 'ref_y_raw','insp_y', 'ref_y',
                                        'csv_name', 'model_predict_right', 'pairs_name']]
        info['dataset'] = datasets
        ref_image, insp_image = model_predict_wrong_df['ref_image'], model_predict_wrong_df['insp_image']

        info['ref_image'] = info['ref_image'].str.replace(cur_color.lower(), target_color.lower(), regex=False)
        info['insp_image'] = info['insp_image'].str.replace(cur_color.lower(), target_color.lower(), regex=False)
        info['ref_image'] = info['ref_image'].str.replace(cur_color.upper(), target_color.upper(), regex=False)
        info['insp_image'] = info['insp_image'].str.replace(cur_color.upper(), target_color.upper(), regex=False)
        for index, info_ in info.iterrows():
            white_ref_image_path = os.path.join(data_path, info_['ref_image'])
            if not os.path.exists(white_ref_image_path):
                continue
            else:
                predict_wrong_lists.append(info.loc[index])
            pass
    # predict_wrong_df = pd.concat(predict_wrong_lists, ignore_index=True)
    predict_wrong_df = pd.DataFrame(predict_wrong_lists)

    predict_wrong_df.to_csv('./predict_wrong_df.csv', index = False)

# def infer_data(datasets, 
#                py_file = "/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/241013_smt_defect_classification/algo_prelabel_singlepinv2_filter_whole_datasets.py"):
#     def run_command(dataset, py_file):
#         command = [
#             "python3", 
#             py_file, 
#             "--valdataset", 
#             dataset
#         ]
#         print(f"processing {dataset}")
#         process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         print(f"processed {dataset}")

#         stdout, stderr = process.communicate()
#         if process.returncode == 0:
#             print(f"Command for {dataset} executed successfully:\n{stdout}")
#         else:
#             print(f"Command for {dataset} failed with error:\n{stderr}")

#     # 使用 ProcessPoolExecutor 并行运行命令
#     with ProcessPoolExecutor(max_workers=len(datasets)) as executor:
#         run_command_partial = partial(run_command, py_file=py_file)
#         executor.map(run_command_partial, datasets)
def run_command(dataset, region, date, version, version_folder, calibration_T, py_file):
    command = [
        "python3", 
        py_file,
        "--region", region,
        "--date", date,
        "--version", version,
        "--version_folder", version_folder, 
        
        "--valdataset", dataset,
        
        "--calibration_T", str(calibration_T)
        

    ]
    print(f"Processing {dataset}", flush=True)
    
    # 使用 subprocess.run 来执行命令并实时输出
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 实时输出 stdout
    for line in process.stdout:
        print(line, end='', flush=True)
    
    # 等待进程结束并获取 stderr
    stderr = process.stderr.read()
    process.wait()
    
    if process.returncode == 0:
        print(f"Command for {dataset} executed successfully.")
    else:
        print(f"Command for {dataset} failed with error:\n{stderr}")

def infer_data(datasets, region, date, version, version_folder, calibration_T, py_file):
    processes = []
    
    for dataset in datasets:
        # 创建进程
        p = multiprocessing.Process(target=run_command, args=(dataset, region, date, version, version_folder, calibration_T, py_file))
        processes.append(p)
        p.start()  # 启动进程

    # 等待所有进程完成
    for p in processes:
        p.join()

def unify_defect_bbtestmz(bbtestmz_csv):
    df = pd.read_csv(bbtestmz_csv, index_col=0)
    df['defect_label'] = df['defect_label'].str.replace('undersolder', 'insufficientsolder')
    df.to_csv(bbtestmz_csv)
    pass
# # 示例调用
# datasets = ['newval', 'bbtest', 'oldval']
# infer_data(datasets, py_file='/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/241013_smt_defect_classification/algo_prelabel_singlepinv2_filter_whole_datasets.py')
if __name__ == "__main__":
    # csv_path = "/mnt/dataset/xianjianming/data_clean/merged_annotation/241022/bbtestmz_pair_purepad_updateaugmented_mz_model_cleaned2.csv"
    # unify_defect_bbtestmz(csv_path)
    # img_folder = "/mnt/dataset/xianjianming/data_clean/"
    # date = "241022"
    # region = "singlepinpad"
    # # vis_defect_pairs(img_folder, date, region, 'train')

    # csv_path = "/mnt/dataset/xianjianming/data_clean/merged_annotation/241022/singlepad_model_clean_changed.csv"
    # data_path = '/mnt/dataset/xianjianming/data_clean/'
    # save_pair_path = './changed_pairs'
    # show_changed_singlepad(csv_path, data_path, save_pair_path)
    # datasets = ['bbtestmz', 'bbtest', 'jiraissues', 'newval', 'alltestval', '3D', 'cur_jiraissue']  #
    # infer_data(datasets, 
    #            'valdataset',
    #            py_file ='/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/241013_smt_defect_classification/algo_prelabel_singlepinv2_filter_whole_datasets.py')
   
    # models = ['singlepadv0.7vs0.9' 'singlepadv0.9tmptselect', 'singlepadv0.6cleaned', 'singlepadv0.9tmptselect2']  #
    # ckp_folder = '/mnt/pvc-nfs-dynamic/robinru/train_tasks/241209_smt_defect_classification/models/checkpoints'
    # version_folders = [
    #     '/singlepadlut0',
    #     '/singlepadlut05',
    #     # '/singlepadlut05_2d',
    #     '/singlepadlut05_modelclean',
    #     # '/singlepadlut05_modelclean_2d',
    #     # '/singlepadlut0_2d',
    #     '/singlepadlut0_modelclean',   2d:'3D', 'cur_jiraissue', 'jiraissues', 'newval', 'alltestval' 'bbtestmz', 'bbtest', 'jiraissues', 'newval', 
    #     # '/singlepadlut0_modelclean_2d',, , 'newval', 'bbtest', 'oldval', 'masked', 'jiraissues', 'cur_jiraissue','cur_jiraissue_26MHZ'
    # ], 'newval', 'bbtest', 'oldval', 'masked', 'jiraissues', 'cur_jiraissue'  , 'jiraissues', 'newval', 'alltestval', '3D', 'cur_jiraissue'
    # datasets =['alltestval', 'newval', 'oldval'] #, '3D', 'cur_jiraissue', 'jiraissues', 'newval', 'bbtestmz', 'bbtest', 
    # infer_data(datasets,  #'alltestval', 'newval', 'bbtest', 'oldval', 'masked', 'jiraissues', 'cur_jiraissue','cur_jiraissue_26MHZ'
    #         'valdataset',
    #         py_file ='/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/241013_smt_defect_classification/algo_prelabel_singlepinv2_cl.py')

    # padgroup
    # rgb:   'alltestval', 'newval', 'bbtest', 'oldval', 'jiraissues', 'masked', 'graphite', 'cur_jiraissue',  'extreme_ng'
    # white: 'alltestval', 'newval', 'bbtest', 'oldval', 'jiraissues', 'cur_jiraissue', 'masked'
    # 
    # datasets = ['alltestval', 'newval', 'graphite', 'body_box_select', 'njzh']  # ,, 'jsd' , 
    # datasets2 = ['oldval', 'jiraissues', 'bbtest', 'masked', 'unacceptable_error', '3D', '3D_supply', 'cur_jiraissue'] 
    # datasets2 = ['njzh'] # , 'newval', 'unacceptable_error', '3D_supply', 'body_box_select', 'njzh'


    # region =  'padgroup'
    # date = '241022'
    # version = 'v1.22.6lut1'
    # inference1 = False
    # inference2 = True
    # calibration_Ts = [1.0]
    # version_folders = [
    #                     # 'padgroupv1.22.6lut1',
    #                     # 'padgroupv1.22.6lut1ed',
    #                     # 'padgroupv1.22.6rgbwhitelut1',
    #                     # 'padgroupv1.22.6rgbwhitelut1ed',
    #                     # 'padgroupv1.22.6rgbwhitelut05diffabs',
    #                     'padgroupv1.22.6lut1select',
    #                     # 'padgroupv1.22.5select',
    #                     # 'padgroupv1.22.6rgbwhitetp1lut05mergerbg',

    #                     # 'padgroupv1.22.5selct',
    #                     # 'padgroupv1.22.6select',
    #                     # 'padgroupv1.22.6',
    #                     # 'padgroupv1.22.6ed',
    #                     # 'padgroupv1.22.6lut05',
    #                     # 'padgroupv1.22.6lut05ed',
    #                     # 'padgroupv1.22.6NGonly',
    #                     # 'padgroupv1.22.6NGonlyed',
    #                     # 'padgroupv1.22.6rgbwhitelut05',
    #                     # 'padgroupv1.22.6rgbwhitelut05ed',

    #                     # 'padgroupv1.22.6rgbwhitelut05cat',
    #                     # 'padgroupv1.22.6rgbwhitelut05catgray',
    #                     # 'padgroupv1.22.6rgbwhitelut05cathwrw',
    #                     # 'padgroupv1.22.6rgbwhitelut05cathwwr',
    #                     # 'padgroupv1.22.6rgbwhitelut05diffrw',
    #                     # 'padgroupv1.22.6rgbwhitelut05diffwr',
    #                     # 'padgroupv1.22.6rgbwhitelut05max',

    #                     # 'padgroupv1.22.6rgbwhitelut05edcat',
    #                     # 'padgroupv1.22.6rgbwhitelut05edcatgray',
    #                     # 'padgroupv1.22.6rgbwhitelut05edcathwrw',
    #                     # 'padgroupv1.22.6rgbwhitelut05edcathwwr',
    #                     # 'padgroupv1.22.6rgbwhitelut05eddiffrw',
    #                     # 'padgroupv1.22.6rgbwhitelut05eddiffwr',
    #                     # 'padgroupv1.22.6rgbwhitelut05edmax',

    #                 ]


    # singlepad
    # rgb: 'bbtestmz', 'bbtest', 'jiraissues', 'newval', 'alltestval', '3D', 'cur_jiraissue', 'black_uuid'
    # datasets = ['bbtestmz', 'bbtest', 'jiraissues', 'newval', 'cur_jiraissue'] # , 'jsd'
    # datasets2 = ['alltestval', '3D_old', '3D', 'yl_cold_solder'] # , 'jsd'

    # inference1 = True
    # inference2 = True
    # region =  'singlepad'
    # version = 'v0.17.0'

    # date = '241022'
    # calibration_Ts = [1.0]
    # version_folders = [     
    #                     'singlepadv0.15.7',

    #                     'singlepadv0.17.0cp05',
    #                     'singlepadv0.17.0cp05clean',
    #                     'singlepadv0.17.0lut05cp05',
    #                     'singlepadv0.17.0lut05cp05clean',
    #                     'singlepadv0.17.0retraincp05',
    #                     'singlepadv0.17.0retraincp05clean',
    #                     'singlepadv0.17.0retrainlut05cp05',
    #                     'singlepadv0.17.0retrainlut05cp05clean',
    #             ]

    # singlepinpad
    # 'alltestval', 'newval', 'bbtestmz', 'jiraissues', 'bbtest',  'cur_jiraissue' , 'hrd_processed_by_bgjc_ori', 'hrd_processed_by_bgjc_aug', 'hrd_processed_by_jm_robin'
    datasets = ['alltestval', 'newval', 'bbtestmz', 'cur_jiraissue']
    datasets2 = ['jiraissues', 'bbtest', '3D'] #, 'cur_jiraissue'
    # datasets = ['zwlb_TG']

    inference1 = True
    inference2 = True
    region =  'singlepinpad'
    version = 'v2.17.7'

    date = '241022'
    calibration_Ts = [1.0]             
    version_folders = [ 
                        'singlepinpadv2.17.3select',
                        'singlepinpadv2.17.7cp05',
                        'singlepinpadv2.17.7cp05clean',
                        'singlepinpadv2.17.7cp05cleantg',
                        'singlepinpadv2.17.7cp05tg',

                        'singlepinpadv2.17.7NGonlycp05',
                        'singlepinpadv2.17.7NGonlycp05clean',
                        'singlepinpadv2.17.7NGonlycp05cleantg',
                        'singlepinpadv2.17.7NGonlycp05tg',
                        

                    ]
    
    # singlepinpadv2.8lut052d, singlepinpadv2.8NGonly2d, singlepinpadv2.8retrain2d, singlepinpadv2.8retrainlut052d, singlepinpadv2.82d
    # singlepinpadv2.8NGonlylut052d, singlepinpadv2.8NGonlyretrain2d, singlepinpadv2.8NGonlyretrainlut052d

    # body 
    # 'jiraissues', 'bbtestmz',  'bbtest', 'newval',  'alltestval', 'cur_jiraissue'
    # 'jiraissues', 'bbtestmz',  'bbtest', 'newval',  'alltestval', 'cur_jiraissue', 'gaoce_led', 'gaoce_trured_over'
    # datasets = ['jiraissues', 'bbtestmz',  'bbtest', 'newval', 'cur_jiraissue']  #'jlc'

    # datasets2 = ['alltestval', 'gaoce_led', 'gaoce_trured_over']  #'jlc'
    # inference1 = True
    # inference2 = True
    # region =  'body'
    # version = 'v2.15.0'

    # date = '241023_white'
    # calibration_Ts = [1.0]

    # version_folders = [
    #                 # 'bodywhiteslimf1v2.14ztsfinalselect',
    #                 'bodywhiteslimf1v2.15.0select',

    #                 # 'bodywhiteslimf1v2.15.0mdbdg',
    #                 # 'bodywhiteslimf1v2.15.0mdbdgretrain',
    #                 # 'bodywhiteslimf1v2.15.0mdbdgts',
    #                 # 'bodywhiteslimf1v2.15.0mdbdgtsretrain',
         
    # ] 

    if inference1:
        for index, version_folder in enumerate(version_folders):
            for c_index, calibration_T in enumerate(calibration_Ts):
                print(f'processing {index}th {version_folder} dir || calibration_T={calibration_T}')
                infer_data(datasets, region, date, version, version_folder, calibration_T,
                    '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/algo_prelabel_singlepinv2_cl.py')

    if inference2:
        for index, version_folder in enumerate(version_folders):
            for c_index, calibration_T in enumerate(calibration_Ts):
                print(f'processing {index}th {version_folder} dir || calibration_T={calibration_T}')
                infer_data(datasets2, region, date, version, version_folder, calibration_T,
                    '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/algo_prelabel_singlepinv2_cl.py')
                    # '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/algo_prelabel_singlepinv2_cl.py')
                

        # infer_data(datasets, version_folder, 
        #            '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/241013_smt_defect_classification/algo_prelabel_singlepinv2_filter_whole_datasets.py')        
        

    # for dataset in datasets:
    #     command = f"""
    #         python3 /mnt/pvc-nfs-dynamic/xianjianming/train_tasks/241013_smt_defect_classification/algo_prelabel_singlepinv2_filter_whole_datasets.py --valdataset {dataset}
    #         """
    #     os.system(command)

    # csv_dir_path = '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/241013_smt_defect_classification/padgroup_ng_white_model_predict_wrong'
    # data_path = '/mnt/dataset/xianjianming/data_clean/'
    # gather_model_predict_wrong(csv_dir_path, data_path, 'white', 'rgb')