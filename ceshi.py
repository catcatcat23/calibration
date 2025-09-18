import os
import random
import pandas as pd

import numpy as np

def analysis_test_on_train_ng_numbers():
    img_folder = '/mnt/pvc-nfs-dynamic/xianjianming/data/merged_annotation/train/data_clean_white/'
    date = '2411'
    region = 'body'
    aug_val_pair_data_filenames = [os.path.join(img_folder,  f'aug_val_pair_labels_{region}_mergedresorted_model_cleaned_250123.csv'),
                                                os.path.join(img_folder,  f'aug_val_pair_labels_{region}_240328b_finalresorted_model_cleaned_250123.csv'),
                                                os.path.join(img_folder,  f'aug_test_pair_labels_{region}_240328b_finalresorted_model_cleaned_250123.csv'),
                                            os.path.join(img_folder, 
                                                        f'aug_val_pair_labels_{region}_merged_withcpmlresorted_model_cleaned_250123.csv'),
                                            os.path.join(img_folder, 
                                                            f'aug_test_pair_labels_{region}_240918sub_final_white_model_cleaned_250123.csv'),
                                            os.path.join(img_folder, 
                                                            f'aug_test_pair_labels_{region}_240919sub_final_white_model_cleaned_250123.csv'),
                                            os.path.join(img_folder, 
                                                        f'aug_test_pair_labels_{region}_241023_final_white_20573_model_cleaned_250123.csv'),
                                            os.path.join(img_folder, 
                                                        f'aug_test_pair_labels_{region}_241023_final_white_20583_model_cleaned_250123.csv'),
    #             os.path.join(img_folder,  f'aug_test_pair_labels_{region}_240819_final_white.csv')
                                                # LED cross pair 241030
                                                os.path.join(img_folder, 
                                                f'LED_cross_pairs_ng_val_model_cleaned_250123.csv'),
                                                os.path.join(img_folder, 
                                                f'LED_cross_pairs_ok_val_model_cleaned_250123.csv'),

                                                # melf cross pair 241031
                                                os.path.join(img_folder, 
                                                f'body_241031_melf_ng_cross_pair2_val_model_cleaned_250123.csv'),
                                                os.path.join(img_folder, 
                                                f'body_241031_melf_ok_cross_pair2_val_model_cleaned_250123.csv'),
                                                os.path.join(img_folder, 
                                                f'aug_test_pair_labels_body_241118_final_white_model_cleaned_250123.csv'),

                                                # os.path.join(img_folder, 
                                                # f'body_white_paris_final_test_refine2.csv'),
                                                #  os.path.join(img_folder, 
                                                # f'body_rgb_paris_final_test_refine2.csv'),
                                                os.path.join(img_folder, 
                                                f'aug_test_pair_labels_body_250122_final_white_DA680_dropoldpairs.csv'),   
                                                os.path.join(img_folder, 
                                                f'aug_test_pair_labels_body_250122_final_white_DA682_dropoldpairs.csv'),
                                                # DA727高测翻转漏报数据，原始数据只有3个缺陷jpair，增强后对train上采样到5，原始train加入到test
                                                # os.path.join(img_folder, 
                                                # f'aug_train_pair_labels_body_250217_final_white_DA727_dropoldpairs.csv'),   
                                                os.path.join(img_folder, 
                                                f'aug_test_pair_labels_body_250217_final_white_DA727_dropoldpairs.csv'),

                                                os.path.join(img_folder, 
                                                f'aug_test_pair_labels_body_250218_final_white_DA727_dropoldpairs.csv'),   
                                                os.path.join(img_folder, 
                                                f'aug_train_pair_labels_body_250218_final_white_DA727_dropoldpairs.csv'),
                                                os.path.join(img_folder, 
                                                                           f'body_white_paris_final_test_refine2.csv')
                                            ]
    
    for csv_path in aug_val_pair_data_filenames:
        cv_name = os.path.basename(csv_path)
        df = pd.read_csv(csv_path)
        df_certain = df[df['confidence'] == 'certain']
        df_ok = df_certain[~df_certain['binary_y']]
        df_ng = df_certain[df_certain['binary_y']]
        print(f"="*30 + f'{cv_name}' + f"="*30)
        print(f'before drop_duplicates total pairs: {len(df_certain)}')
        print(f'before drop_duplicates total ok pairs: {len(df_ok)}')
        print(f'before drop_duplicates total ng pairs: {len(df_ng)}')


        df_certain_drop = df_certain.drop_duplicates(['ref_image', 'insp_image'])
        df_drop_ok = df_certain_drop[~df_certain_drop['binary_y']]
        df_drop_ng = df_certain_drop[df_certain_drop['binary_y']]
        print(f"*"*30 + f'after drop_duplicates' + f"*"*30)
        print(f'after drop_duplicates total pairs: {len(df_certain_drop)}')
        print(f'after drop_duplicates total ok pairs: {len(df_drop_ok)}')
        print(f'after drop_duplicates total ng pairs: {len(df_drop_ng)} \n')



def analysis_train_ng_numbers():
    img_folder = '/mnt/pvc-nfs-dynamic/xianjianming/data/merged_annotation/train/data_clean_white/'
    date = '2411'
    region = 'body'
    aug_train_pair_data_filenames = [os.path.join(img_folder, 
                                                      f'aug_train_pair_labels_{region}_mergedresorted_model_cleaned_250123.csv'),
                                         os.path.join(img_folder, 
                                                      f'aug_train_pair_labels_{region}_240328b_finalresorted_model_cleaned_250123.csv'),
                                             os.path.join(img_folder, 
                                                          f'aug_train_selfpair_labels_{region}_240328b_finalresorted_model_cleaned_250123.csv'),
                                             os.path.join(img_folder, 
                                                        f'aug_train_pair_labels_{region}_merged_withcpmlresorted_model_cleaned_250123.csv'),
                                             os.path.join(img_folder, 
                                                          f'aug_train_pair_labels_{region}_240725_final_white_model_cleaned_250123.csv'),
#                                             os.path.join(img_folder, 
#                                                          f'aug_train_pair_labels_{region}_240913_final_white.csv'),
                                             os.path.join(img_folder, 
                                                         f'aug_train_pair_labels_{region}_240918_final_white_model_cleaned_250123.csv'),
                                             os.path.join(img_folder, 
                                                         f'aug_train_pair_labels_{region}_240919_final_white_model_cleaned_250123.csv'),
                                             os.path.join(img_folder, 
                                                         f'aug_train_pair_labels_{region}_240926sub_final_white_model_cleaned_250123.csv'),
                                             os.path.join(img_folder, 
                                                      f'aug_train_pair_labels_{region}_241023_final_white_20573_model_cleaned_250123.csv'),
                                             os.path.join(img_folder, 
                                                      f'aug_train_pair_labels_{region}_241023_final_white_20583_model_cleaned_250123.csv'),
                                             os.path.join(img_folder, 
                                                      f'aug_train_pair_labels_{region}_241023_final_white_240071_model_cleaned_250123.csv'),

                                             # LED cross pair 241030
                                            os.path.join(img_folder, 
                                            f'LED_cross_pairs_ok_train_model_cleaned_250123.csv'),
                                            os.path.join(img_folder, 
                                            f'LED_cross_pairs_ng_train_model_cleaned_250123.csv'),                                         

                                            #  melf cross pair 241031
                                            os.path.join(img_folder, 
                                            f'body_241031_melf_ng_cross_pair2_train_model_cleaned_250123.csv'),
                                            os.path.join(img_folder, 
                                            f'body_241031_melf_ok_cross_pair2_train_model_cleaned_250123.csv'),
                                            os.path.join(img_folder, 
                                            f'aug_train_pair_labels_body_241118_final_white_model_cleaned_250123.csv'),
     

                                            os.path.join(img_folder, 
                                            f'aug_train_pair_labels_body_250122_final_white_DA680_dropoldpairs.csv'),
                                            os.path.join(img_folder, 
                                            f'corss_ok_pair_DA680.csv'),
                                            os.path.join(img_folder, 
                                            f'aug_corss_ng_pair_DA680.csv'),
                                            os.path.join(img_folder, 
                                            f'aug_train_pair_labels_body_250122_final_white_DA682_dropoldpairs.csv'),   
                                            # DA727高测翻转漏报数据，原始数据只有3个缺陷jpair，增强后对train上采样到5，原始train加入到test
                                            # os.path.join(img_folder, 
                                            #  f'aug_train_pair_labels_body_250217_final_white_DA727_dropoldpairs_up.csv'),  
                                            os.path.join(img_folder, 
                                             f'aug_train_pair_labels_body_250217_final_white_DA727_dropoldpairs.csv'), 
                                            os.path.join(img_folder, 
                                             f'aug_train_pair_labels_body_250218_final_white_DA727_dropoldpairs_ok_cross_pair.csv'),  
                                            os.path.join(img_folder, 
                                             f'aug_train_pair_labels_body_250218_final_white_DA727_dropoldpairs_ng_cross_pair.csv'), 
                                             os.path.join(img_folder, 
                                                         f'aug_train_pair_labels_{region}_240913_final_white_model_cleaned_250123.csv'),
                                            os.path.join(img_folder, 
                                                         f'body_white_paris_final_train_refine2.csv'),
                                              
            ]

    
    for csv_path in aug_train_pair_data_filenames:
        cv_name = os.path.basename(csv_path)
        df = pd.read_csv(csv_path)
        df_certain = df[df['confidence'] == 'certain']
        df_ok = df_certain[~df_certain['binary_y']]
        df_ng = df_certain[df_certain['binary_y']]
        print(f"="*30 + f'{cv_name}' + f"="*30)
        print(f'before drop_duplicates total pairs: {len(df_certain)}')
        print(f'before drop_duplicates total ok pairs: {len(df_ok)}')
        print(f'before drop_duplicates total ng pairs: {len(df_ng)}')


        df_certain_drop = df_certain.drop_duplicates(['ref_image', 'insp_image'])
        df_drop_ok = df_certain_drop[~df_certain_drop['binary_y']]
        df_drop_ng = df_certain_drop[df_certain_drop['binary_y']]
        print(f"*"*30 + f'after drop_duplicates' + f"*"*30)
        print(f'after drop_duplicates total pairs: {len(df_certain_drop)}')
        print(f'after drop_duplicates total ok pairs: {len(df_drop_ok)}')
        print(f'after drop_duplicates total ng pairs: {len(df_drop_ng)} \n')

def get_singlepad_black_uuid(img_foler, region, date):
    component_csv = '/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/component-export-2025-03-04-20-47-54.csv'
    annotation_folder = os.path.join(img_foler + 'merged_annotation', 'all_singlepad_pinpad_csv')
    aug_train_pair_data_filenames = [
                                                 os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250103_final_rgb_DA620_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_test_pair_labels_singlepad_250103_final_rgb_DA620_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv'),
                                         os.path.join(annotation_folder,
                                                     f'aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv'),      
                                            os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs_filter.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_test_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs_filter.csv'),
                                         os.path.join(annotation_folder,
                                                     f'aug_test_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv'),    
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_{region}_nonpaired_rgb_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_{region}_merged_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_{region}_240329_final_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_{region}_240428_final_RGB_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_{region}_240429_final_RGB_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_{region}_240507_final_RGB_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_{region}_240808_final_RGB_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_{region}_241018_final_rgb_model_cleaned.csv'),
                                        # 241111 241114
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241103_final_rgb_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241111_final_rgb_D433_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241114_final_rgb_DA472_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241114_final_rgb_DA465_update_241205_model_cleaned.csv'),

                                        # 241126          
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241126_final_rgb_DA505_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241126_final_rgb_DA507_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241127_final_rgb_DA509_model_cleaned.csv'),

                                        # 241209          
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241206_final_rgb_DA534.csv'),

                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241209_final_rgb_DAJIRA_1367.csv'),         

                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241206_final_rgb_DA519.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs.csv'),         

                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_250110_final_rgb_DA629-2_dropoldpairs.csv'),
                    
                                        os.path.join(annotation_folder,f'aug_val_pair_labels_{region}_merged_model_cleaned.csv'),
                                        os.path.join(annotation_folder, f'aug_val_pair_labels_{region}_240329_final_model_cleaned.csv'),
                                        os.path.join(annotation_folder, f'aug_test_pair_labels_{region}_240329_final_model_cleaned.csv'),

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_{region}_240428_final_RGB_model_cleaned.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_{region}_240429_final_RGB_model_cleaned.csv'),
                                       os.path.join(annotation_folder,f'aug_val_pair_labels_{region}_240808_final_RGB_model_cleaned.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_{region}_241018_final_rgb_model_cleaned.csv'),
                                       # 3D lighting  
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241103_final_rgb_model_cleaned.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241111_final_rgb_D433_model_cleaned.csv'),

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241114_final_rgb_DA472_model_cleaned.csv'),

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241126_final_rgb_DA505_model_cleaned.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241126_final_rgb_DA507_model_cleaned.csv'),

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241206_final_rgb_DA534.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241206_final_rgb_DA519.csv'),             
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250110_final_rgb_DA629-2_dropoldpairs.csv'),
                                         os.path.join(annotation_folder, 
                                            f'aug_val_pair_labels_singlepad_merged_model_cleaned2.csv'),
                                          os.path.join(annotation_folder, 
                                            f'bbtest_labels_pairs_singlepad_230306v2_model_cleaned2.csv'),                    
                                          os.path.join(annotation_folder, 
                                            f'aug_val_pair_labels_singlepad_240329_final_model_cleaned2.csv'),
                                          os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_240329_final_model_cleaned2.csv'),
                                          os.path.join(annotation_folder, 
                                            f'jiraissues_pair_labels_singlepad_240314_final_model_cleaned2.csv'),
                                          os.path.join(annotation_folder, 
                                            f'test_pair_labels_singlepad_240403debug_final_model_cleaned2.csv'),
                                          os.path.join(annotation_folder, 
                                            f'test_pair_labels_singlepad_240404debug_final_model_cleaned2.csv'),
        
                                          os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_240428_final_RGB_model_cleaned2.csv'),
            
                                            os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_240429_final_RGB_model_cleaned2.csv'),
                                            os.path.join(annotation_folder, 
                                            f'jira_test_pair_labels_singlepad_240429_model_cleaned2.csv'),
                                            os.path.join(annotation_folder, 
                                            f'aug_train_pair_labels_singlepad_240507_final_RGB_model_cleaned2.csv'),
                                             os.path.join(annotation_folder, 
                                            f'bbtestmz_pair_purepad_updateaugmented_mz_model_cleaned2.csv'),
                                            # os.path.join(annotation_folder, 
                                            # f'bbtestmz_pair_purepad_input_update_model_cleaned2.csv'),
   
                                            os.path.join(annotation_folder, 
                                            f'aug_val_pair_labels_singlepad_240808_final_RGB_model_cleaned2.csv'),
                                             os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_241018_final_rgb_model_cleaned2.csv'),
                                            os.path.join(annotation_folder, 
                                            f'aug_train_pair_labels_singlepad_241018_final_rgb_model_cleaned2.csv'),

                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_241103_final_rgb_model_cleaned2.csv'),
                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_241111_final_rgb_D433_model_cleaned2.csv'),
                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_241114_final_rgb_DA465_model_cleaned2.csv'),
                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_241114_final_rgb_DA472_model_cleaned2.csv'),

                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_241126_final_rgb_DA505_model_cleaned2.csv'),
                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_241126_final_rgb_DA507_model_cleaned2.csv'),

                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_241127_final_rgb_DA509.csv'),
                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_241206_final_rgb_DA534.csv'),
                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_241206_final_rgb_DA519.csv'),
        

                                            os.path.join(annotation_folder, 
                                                      f'aug_test_pair_labels_singlepad_250103_final_rgb_DA620_dropoldpairs.csv') ,     
                                            os.path.join(annotation_folder, 
                                                    f'aug_test_pair_labels_singlepad_D620_SZGC_sp.csv'),  

                                            os.path.join(annotation_folder, 
                                            f'aug_train_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs_ori.csv') ,   
                                            os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs_ori.csv') ,

                                            os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs.csv'),
                                            os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_250110_final_rgb_DA629-2_dropoldpairs.csv'),                        
        

                                            os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs.csv'),
                                            os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv'),  
        
                                            os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_250213_final_rgb_DA712_dropoldpairs.csv'),
                                            os.path.join(annotation_folder, 
                                            f'aug_train_pair_labels_singlepad_250213_final_rgb_DA712_dropoldpairs.csv'),   
        
                                            os.path.join(annotation_folder, 
                                            f'aug_test_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs.csv'),
                                            os.path.join(annotation_folder, 
                                            f'aug_train_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs.csv') ,

                                       ]
 
    all_csv = list(set(aug_train_pair_data_filenames))

    exists_df = [pd.read_csv(f) for f in all_csv]

    all_df = pd.concat(exists_df, ignore_index=True)
    component_df = pd.read_csv(component_csv)

    # 创建布尔掩码，筛选出 a 和 b 都不在 all_df 的 a 列中的行
    mask = ~component_df['uuid'].isin(all_df['ref_uuid']) & ~component_df['uuid'].isin(all_df['insp_uuid'])

    # 应用掩码筛选数据
    filtered_df = component_df[mask].reset_index(drop=True)
    filtered_df.to_csv('/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/component-export-2025-03-04-20-47-54-singlepad-filter.csv')


def get_singlepinpad_black_uuid(img_foler, region, date):
    component_csv = '/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/component-export-2025-03-04-20-47-54.csv'
    annotation_folder = os.path.join(img_foler + 'merged_annotation', 'all_singlepad_pinpad_csv')
    aug_train_pair_data_filenames = [
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_{region}_merged.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_{region}_240329_final_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_{region}_240424_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_{region}_240428_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_{region}_240429_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240715_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240702_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240708_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240725_final_rgb_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240930_final_rgbv2_model_cleaned.csv'),
                                         # 3D lighting                                       
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241018_final_rgb_model_cleaned.csv'),

                                         # 241111, 241114
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241111_final_rgb_D433_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241114_final_rgb_DA472_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241114_final_rgb_DA465_model_cleaned.csv'),

                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241206_final_rgb_DA534_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241206_final_rgb_DA519_model_cleaned.csv'),

                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241220_final_rgb_DA1620_dropoldpairs.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241231_final_rgb_DA3031_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_250121_final_rgb_DA677_dropoldpairs.csv'),
                                                                                    
            
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_ng_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_240913_final_rgb_ng_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_240919_final_rgb_ng_model_cleaned.csv'),

                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_240913_final_rgb_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_240919_final_rgb_model_cleaned.csv'),

                                        os.path.join(annotation_folder,
                                                     f'aug_val_pair_labels_{region}_merged_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_val_pair_labels_{region}_240329_final_model_cleaned.csv'),
                                       os.path.join(annotation_folder, f'aug_test_pair_labels_{region}_240329_final_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_240424_final_RGB_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_240428_final_RGB_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_240429_final_RGB_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_240716_final_RGB_model_cleaned.csv'),
                                       # 3D lighting                                 
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_241018_final_rgb_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241111_final_rgb_D433_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241114_final_rgb_DA472_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241206_final_rgb_DA534_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241206_final_rgb_DA519_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241220_final_rgb_DA1620_dropoldpairs.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241231_final_rgb_DA3031_dropoldpairs.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250121_final_rgb_DA677_dropoldpairs.csv'),  


                                         os.path.join(annotation_folder,
                                            f'bbtestmz_pair_pinpad_input_update_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'bbtestmz_pair_pinpad_updateaugmented_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                            f'aug_val_pair_labels_singlepinpad_merged_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'bbtest_labels_pairs_singlepinpad_230306v2_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'aug_val_pair_labels_singlepinpad_240329_final_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_singlepinpad_240329_final_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'jiraissues_pair_labels_singlepinpad_240314_final_model_cleaned.csv'),
#                                       val_image_pair_path6 = os.path.join(annotation_folder,
#                                             f'test_pair_labels_singlepinpad_240403debug_final.csv')
                                        #  os.path.join(annotation_folder,
                                        #     f'debug_singlepinpad_240816_update_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'test_pair_labels_singlepinpad_240404debug_final_model_cleaned.csv'),
        
        
                                         os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_singlepinpad_240424_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_singlepinpad_240428_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_singlepinpad_240429_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'jira_test_pair_labels_singlepinpad_240429_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_singlepinpad_240716_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                            f'aug_train_pair_labels_singlepinpad_240913_final_rgb_model_cleaned.csv'),

                                             os.path.join(annotation_folder,
                                            f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_model_cleaned.csv'),

                                             os.path.join(annotation_folder,
                                            f'aug_train_pair_labels_singlepinpad_240919_final_rgb_ng_model_cleaned.csv'),
                                             os.path.join(annotation_folder,
                                            f'aug_train_pair_labels_singlepinpad_240930_final_rgbv2_model_cleaned.csv'),

                                             os.path.join(annotation_folder,
                                             f'aug_train_pair_labels_singlepinpad_241018_final_rgb_model_cleaned.csv'),
                                             os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_singlepinpad_241018_final_rgb_model_cleaned.csv'),
        
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241111_final_rgb_D433_model_cleaned.csv'),
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241114_final_rgb_DA472_model_cleaned.csv'),
                                            os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241206_final_rgb_DA534_model_cleaned.csv'),
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241206_final_rgb_DA519_model_cleaned.csv'),
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241220_final_rgb_DA1620_dropoldpairs.csv'),
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241231_final_rgb_DA3031_dropoldpairs.csv'),

                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250109_final_rgb_DA627-629_dropoldpairs_ori.csv'),
                                             os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250109_final_rgb_DA627-629_dropoldpairs_ori.csv')   ,     

                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'),
        
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),  
         
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv'), 
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv') ,
                                            os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv') ,
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv')  ,                         
        
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv'),
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250220_final_rgb_DA730_supply_dropoldpairs.csv')  , 
                                             os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250220_final_rgb_DA730_supply_dropoldpairs.csv') ,
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250222_final_rgb_DA743_dropoldpairs.csv') ,  
                                             os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250222_final_rgb_DA743_dropoldpairs.csv') ,
                                             os.path.join(annotation_folder,
                                                    f'aug_black_test_pair_labels_singlepinpad_250219_final_rgb_DA730_45_dropoldpairs_cross_pair.csv'),

                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_just_test.csv'),

                                             os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_just_test.csv'),
        
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),

                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),
    
                                             os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv'),
        

        
                                             os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_singlepinpad_250222_final_rgb_DA743_dropoldpairs.csv') ,  
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                      os.path.join(annotation_folder,'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv'),
                      os.path.join(annotation_folder,'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_supply.csv'),
                      os.path.join(annotation_folder,'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv'),
                      os.path.join(annotation_folder,'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                      os.path.join(annotation_folder,'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv'),
                      os.path.join(annotation_folder,'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv'),
                      os.path.join(annotation_folder,'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_supply.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                      os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),                      
                      os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),                    
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_supply.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),                      
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_supply.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs.csv'),   
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250213_final_rgb_DA712_dropoldpairs_cross_pair.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250213_final_rgb_DA712_dropoldpairs.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs_filter.csv'),
                    os.path.join(annotation_folder, 'aug_train_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs_filter.csv'),
                    os.path.join(annotation_folder, 'aug_test_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv'),
   
    ]
 
    all_csv = list(set(aug_train_pair_data_filenames))

    exists_df = [pd.read_csv(f) for f in all_csv]

    all_df = pd.concat(exists_df, ignore_index=True)
    component_df = pd.read_csv(component_csv, index_col=0)

    # 创建布尔掩码，筛选出 a 和 b 都不在 all_df 的 a 列中的行
    mask = ~component_df['uuid'].isin(all_df['ref_uuid']) & ~component_df['uuid'].isin(all_df['insp_uuid'])

    # 应用掩码筛选数据
    filtered_df = component_df[mask].reset_index(drop=True)
    filtered_df.to_csv('/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/component-export-2025-03-04-20-47-54-singlepinpad-filter.csv')


def getout_singlepinpad_pad_inter():
    singlepad_csv = '/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/component-export-2025-03-04-20-47-54-singlepad-filter.csv'
    singlepinpad_csv = '/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/component-export-2025-03-04-20-47-54-singlepinpad-filter.csv'

    singlepad_df = pd.read_csv(singlepad_csv, index_col=0)
    singlepinpad_df = pd.read_csv(singlepinpad_csv, index_col=0)

    common_a_values = set(singlepad_df['uuid']).intersection(set(singlepinpad_df['uuid']))
    filtered_singlepad_df = singlepad_df[singlepad_df['uuid'].isin(common_a_values)]
    
    filtered_singlepad_df = filtered_singlepad_df.reset_index(drop=True)
    filtered_singlepad_df.to_csv('/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/component-export-2025-03-04-20-47-54-filter.csv')

def rename_bash_date(bash_dir, version_list, replace_s, new_s):

    for verison_dir in version_list:
        for bash_name in os.listdir(os.path.join(bash_dir, verison_dir)):
            if '.sh' not in bash_name:
                continue
            new_bash_name = bash_name.replace(replace_s, new_s)

            old_bash_path = os.path.join(bash_dir, verison_dir, bash_name)
            new_bash_path = os.path.join(bash_dir, verison_dir, new_bash_name)

            os.system(f"sudo mv {old_bash_path} {new_bash_path}")

def temp(annotation_folder, dest_folder):
       aug_train_pair_data_filenames = [
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_model_cleaned_250309.csv'),
        

                                                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240930_final_rgbv2_model_cleaned_250309.csv'),
                                         # 3D lighting                                       
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241018_final_rgb_model_cleaned_250309.csv'),
            ]
       aug_train_pair_data_filenames += [

                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240913_final_rgb_model_cleaned_250309.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240919_final_rgb_ng_model_cleaned_250309.csv'),
            ]

       for file_path in aug_train_pair_data_filenames:
           
           if os.path.exists(file_path):

              file_name = os.path.basename(file_path)
              dest_path = os.path.join(dest_folder, file_name)

              os.system(f'sudo cp {file_path} {dest_path}')
           else:
               print(f"no || {file_path}")
from glob import glob
from PIL import Image
from tqdm import tqdm
import shutil
def zip_body(ori_dir, save_dir):
    all_images = []
    for root, _, files in os.walk(ori_dir):
        for file in files:

            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 可加.jpeg
                full_path = os.path.join(root, file)
                all_images.append(full_path)
    
    for img_path in  tqdm(all_images):
        dst_img = img_path.replace(ori_dir, save_dir)
        dst_dir = os.path.dirname(dst_img)
        os.makedirs(dst_dir, exist_ok=True)

        if file.lower().endswith(('.jpg', '.jpeg')):
            shutil.copy2(img_path, dst_img)
            # os.system(f'sudo cp {img_path} {dst_img}')
        else:
            dst_jpg_img = dst_img.replace('.png', '.jpg')
            # dst_dir = os.path.dirname(dst_jpg_img)
            # os.makedirs(dst_dir,exist_ok=True)
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")  # PNG可能带透明通道
                rgb_img.save(dst_jpg_img, "JPEG")
                
    os.system(f'sudo cd {save_dir} && sudo zip -r body_jpg_all.zip body_jpg_all')

def down_sehuan_data(sehuang_csv, data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sh_df = pd.read_csv(sehuang_csv, index_col=0)
    sh_df = sh_df[sh_df['confidence']== 'certain']
    sh_unique_df = sh_df.drop_duplicates(['ref_image', 'insp_image'])
    # 合并两列为一列（纵向堆叠）
    image_path_series = pd.concat([sh_unique_df['ref_image'], sh_unique_df['insp_image']], ignore_index=True)
    
    # 转为 DataFrame
    image_path_df = pd.DataFrame({'image_path': image_path_series})
    
    sh_unique_df = image_path_df.drop_duplicates(['image_path'])

    for index, infos in sh_unique_df.iterrows():
        image_path = os.path.join(data_dir, infos['image_path'])
        image_name = os.path.basename(infos['image_path'])
        dst_path = os.path.join(save_dir, image_name)
        if os.path.exists(image_path):
            os.system(f"sudo cp {image_path} {dst_path}")

import cv2
from dataloader.image_resampler_pins import LUT_DEBUG
def vis_lut(img_dir, save_dir, suffix = 'png'):
    os.makedirs(save_dir, exist_ok=True)
    lut = LUT_DEBUG(lut_path="/mnt/pvc-nfs-dynamic/xianjianming/data/merged_annotation/insp.lut.png", lut_p=1)
    for img_path in glob(os.path.join(img_dir, f"*.{suffix}")):
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)
        print(f"processing {img_name}")
        save_img_path = os.path.join(save_dir, img_name.replace(f'.{suffix}', f'_lut.{suffix}'))
        lut_img = lut(img, save_img_path)

def cp_rename_bash(bash_dir, suffix):
    for bash_path in glob(os.path.join(bash_dir, "*.sh")):
        new_bash_path = bash_path.replace('.sh', f'_{suffix}.sh')
        os.system(f"sudo cp {bash_path} {new_bash_path}")

if __name__ == "__main__":
    bash_dir = '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/bash_scripts/singlepinpad/250814'
    suffix = 'lr'
    # cp_rename_bash(bash_dir, suffix)
    sehuang_csv = '/mnt/dataset/xianjianming/data_clean/merged_annotation/241022/sehuang.csv'
    data_dir = '/mnt/dataset/xianjianming/data_clean_white/'
    save_dir = '/mnt/dataset/xianjianming/data_clean_white/sehuang'
    # down_sehuan_data(sehuang_csv, data_dir, save_dir)

    ori_dir = '/mnt/dataset/xianjianming/data_clean_white'
    save_dir = '/mnt/pvc-nfs-dynamic/body_jpg_all'
#     os.makedirs(save_dir,exist_ok=True)
#     zip_body(ori_dir, save_dir)

    # analysis_test_on_train_ng_numbers()
    # analysis_train_ng_numbers()

    annotation_folder = '/mnt/dataset/xianjianming/data_clean/merged_annotation/241022/'
    dest_folder = '/mnt/pvc-nfs-dynamic/xianjianming/data/merged_annotation/train/data_clean/merged_annotation/2501/'
       # temp(annotation_folder, dest_folder)

    bash_dir =  '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/bash_scripts/singlepad'
    version_list =  ['250820_lut0_05_seed42']
    replace_s =  '_250812'
    new_s = '_250820_lut0_05_seed42'
    rename_bash_date(bash_dir, version_list, replace_s, new_s)

    img_dir = '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/esd_4'
    save_dir = '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/esd_4_lut_vis'

#     vis_lut(img_dir, save_dir, suffix = 'jpg')
    # import torch
    # print(torch.backends.cudnn.version())


#     get_singlepad_black_uuid('/mnt/dataset/xianjianming/data_clean/', 'singlepad', 'all_singlepad_pinpad_csv')
#     get_singlepinpad_black_uuid('/mnt/dataset/xianjianming/data_clean/', 'singlepinpad', 'all_singlepad_pinpad_csv')
#     getout_singlepinpad_pad_inter()
    pass
