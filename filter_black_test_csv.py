import os

import pandas as pd

from glob import glob


def get_singlepad_black_uuid(img_foler, region):
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
    # all_df.to_csv('/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/all_singlepad.csv')
    component_df = pd.read_csv(component_csv)

    mask = ~component_df['uuid'].isin(all_df['ref_uuid']) & ~component_df['uuid'].isin(all_df['insp_uuid'])

    filtered_df = component_df[mask].reset_index(drop=True)
    filtered_df.to_csv('/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/component-export-2025-03-04-20-47-54-singlepad-filter.csv')


def get_singlepinpad_black_uuid(img_foler, region):
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
    all_df.to_csv('/mnt/dataset/xianjianming/data_clean/merged_annotation/all_singlepad_pinpad_csv/all_singlepinpad.csv')

    component_df = pd.read_csv(component_csv, index_col=0)

    mask = ~component_df['uuid'].isin(all_df['ref_uuid']) & ~component_df['uuid'].isin(all_df['insp_uuid'])

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


def filter_singlepad_black_test_csv(annotation_folder, part_name, save_path):
        if part_name == 'singlepad':
            test_annotation_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_singlepad_230306v2_model_cleaned2.csv')]
            aug_train_pair_data_filenames = [
                                        # 金赛点误报筛选后数据
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs.csv'),            
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_nonpaired_rgb_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_merged_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_240329_final_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_240428_final_RGB_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_240429_final_RGB_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_240507_final_RGB_model_cleaned2.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepad_240808_final_RGB_model_cleaned.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_241018_final_rgb_model_cleaned2.csv'),
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
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepad_general_final_rgb_hrd_ks_dropoldpairs_clean2.csv'),                                                      
                    
                                         ]
       
            aug_val_pair_data_filenames = [
                                        os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,f'aug_val_pair_labels_singlepad_merged_model_cleaned2.csv'),
                                        os.path.join(annotation_folder, f'aug_val_pair_labels_singlepad_240329_final_model_cleaned2.csv'),
                                        os.path.join(annotation_folder, f'aug_test_pair_labels_singlepad_240329_final_model_cleaned2.csv'),

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_240428_final_RGB_model_cleaned2.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_240429_final_RGB_model_cleaned2.csv'),
                                       os.path.join(annotation_folder,f'aug_val_pair_labels_singlepad_240808_final_RGB_model_cleaned2.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241018_final_rgb_model_cleaned2.csv'),
                                       # 3D lighting  
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241103_final_rgb_model_cleaned2.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241111_final_rgb_D433_model_cleaned2.csv'),
                                    #    os.path.join(annotation_folder,
                                    #                   f'aug_test_pair_labels_singlepad_241114_final_rgb_DA465.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241114_final_rgb_DA472_model_cleaned2.csv'),

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241126_final_rgb_DA505_model_cleaned2.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241126_final_rgb_DA507_model_cleaned2.csv'),

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241206_final_rgb_DA534.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241206_final_rgb_DA519.csv'),             
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250110_final_rgb_DA629-2_dropoldpairs.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_general_final_rgb_hrd_ks_dropoldpairs_clean2.csv'), 
                                       ]
            test_all_csv = [
                # 深圳南方虚焊分数过低
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250327_final_rgb_DA825_op_dropoldpairs.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250327_final_rgb_DA825_op_dropoldpairs.csv'),
                # 辉瑞大偏移漏报
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250324_final_rgb_DA816_op_dropoldpairs.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250317_final_rgb_DA780_jsd_fp_dropoldpairs.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250317_final_rgb_DA780_jsd_fp_dropoldpairs.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs_TG.csv'),

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
                                                    f'bbtestmz_pair_purepad_updateaugmented_mz_model_cleaned_250310.csv'),
                # val_image_pair_path13 : os.path.join(annotation_folder,
                #                                     f'bbtestmz_pair_purepad_input_update_model_cleaned2.csv')
                # val_image_pair_path14 : os.path.join(annotation_folder,
                #                                     f'aug_test_pair_labels_singlepad_240716_final_RGB_model_cleaned2.csv')
                 os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_singlepad_240808_final_RGB_model_cleaned2.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_241018_final_rgb_model_cleaned2.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_241018_final_rgb_model_cleaned2.csv'),
                # 3D
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
                #############
                # 处理后的典烨数据，可加入一般测试
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250110_final_rgb_DA629-2_dropoldpairs.csv')                        ,
                
                # DA703昆山启佳特供数据，仅用于cur_jiraissue
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv')  ,
                
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250213_final_rgb_DA712_dropoldpairs.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250213_final_rgb_DA712_dropoldpairs.csv')   ,
                
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs.csv'),
                 os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs.csv') ,

                # 昆山启佳通用数据
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_general_final_rgb_hrd_ks_dropoldpairs_clean2.csv') ,

                # 黑盒uuid测试集(训练集测试集都用于测试)
                 os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250310_final_rgb_DA758_black_uuid_dropoldpairs.csv') ,
                 os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250310_final_rgb_DA758_black_uuid_dropoldpairs.csv') ,
              ]
            train_relative_csv = test_annotation_filenames + aug_train_pair_data_filenames + aug_val_pair_data_filenames

            tmp_list= []
            [tmp_list.append(x) for x in train_relative_csv  if x not in tmp_list]
            train_relative_csv = deepcopy(tmp_list)

            tmp_list= []
            [tmp_list.append(x) for x in test_all_csv  if x not in tmp_list]
            test_all_csv = deepcopy(tmp_list)
            test_all_csv = [                 os.path.join(annotation_folder,
                                                    f'all_singlepad_rgb_uncertain_before_pos.csv'),]
        elif part_name == 'singlepinpad':
            pass

        train_df = parse_csv_lists(train_relative_csv, 'certain')
        train_uncertain_df = parse_csv_lists(train_relative_csv, 'uncertain')
        test_all_df = parse_csv_lists(test_all_csv, None)
       
        # candidate_black_df = pd.concat([train_uncertain_df, test_all_df], ignore_index=True).drop_duplicates(['ref_image', 'insp_image'])
        all_train = pd.concat([train_uncertain_df, train_df], ignore_index=True).drop_duplicates(['ref_image', 'insp_image'])

        candidate_black_df = test_all_df
        # on_train_imgs = set(train_df["ref_image"]).union(set(train_df["insp_image"]))
        on_train_imgs = set(all_train["ref_image"]).union(set(all_train["insp_image"]))

        black_test_imgs_df = candidate_black_df[~candidate_black_df["ref_image"].isin(on_train_imgs) & ~candidate_black_df["insp_image"].isin(on_train_imgs)]
        black_test_imgs_df.to_csv(save_path)

        # black_test_cataforys = black_test_imgs_df.groupby(['insp_defect_label'])
        # print(black_test_cataforys.groups.keys())
        # black_test_cataforys.size()



def parse_csv_lists(csv_lists, confidence = None): 
    df_list = [pd.read_csv(x) for x in csv_lists]
    df = pd.concat(df_list, ignore_index=True)

    if confidence is not None:
       
       target_df = df[df['confidence'] == confidence]
    else:
       target_df = df
    
    assert not target_df['ref_image'].isna().any(), '金板图不能为空'
    assert not target_df['insp_image'].isna().any(), '检测图不能为空'

    target_df = target_df.drop_duplicates(['ref_image', 'insp_image'])
    return target_df

if __name__ == "__main__":
     from copy import deepcopy
     save_path = './all_singlepad_rgb_uncertain_before_pos_black_test.csv'
     filter_singlepad_black_test_csv('/mnt/dataset/xianjianming/data_clean/data_clean_all_csv_250402', 'singlepad', save_path)
    # get_singlepad_black_uuid('/mnt/dataset/xianjianming/data_clean/', 'singlepad')
#     get_singlepinpad_black_uuid('/mnt/dataset/xianjianming/data_clean/', 'singlepinpad')
    # getout_singlepinpad_pad_inter()

     pass