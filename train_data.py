import os

def get_train_csv(annotation_folder, region, version_name, special_data = None):
    
    if region == 'singlepinpad':
        # annotation_filename = os.path.join(annotation_folder, f'trainval_labels_singlepinpad.csv')
        annotation_filename = None

        val_annotation_filename = None
        test_annotation_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_singlepinpad_230306v2_model_cleaned.csv')]

        aug_train_pair_data_filenames = [
                                        #深圳明弘达返回虚焊未检出250825
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_ng_corss_pair_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_ok_corss_pair_up2_250828.csv'),
                                        # 福建泉州智领返回的虚焊
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_ok_corss_pair2_up.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_ok_corss_pair2_up.csv'),
                                                                                                                           
    
                                        # 广州华创冷焊未检出 #TODO 该部分虚焊数据很重要，需要重新整理一下
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepinpad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepinpad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs_ok_corss_pair_up.csv'),
                                        
                                        # 广州华创虚焊未检出
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                                        
                                        # 深圳明弘达返回虚焊未检出
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_ok_cross_pair_clean_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_ng_cross_pair_clean.csv'),
                                                                                

                                        # 江苏昆山丘钛翘脚虚焊漏报
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250707_final_rgb_DA1341_jxks_xh_op_dropoldpairs_clean.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250707_final_rgb_DA1341_jxks_xh_op_dropoldpairs_clean_ng_cross_pair_clean_up.csv'),
                                        # 人工制作翘脚
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepinpad_250710_final_rgb_DA1388_jsks_xh_op_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepinpad_250710_final_rgb_DA1388_jsks_xh_op_dropoldpairs_ng_cross_pair_clean_up.csv'),

                                        # 深圳达人高科翘脚漏报
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250627_final_rgb_DA1292_szdrgk_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250627_final_rgb_DA1292_szdrgk_dropoldpairs_ok_cross_pair_up.csv'),
                                        # 澜悦翘脚漏报
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250620_final_rgb_DA1241_ly_qj_dropoldpairs.csv'),
                                        
                                        # 永林冷焊
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_ok_cross_pair_up.csv'),
                                                                                      
                                    # 深圳富洛翘脚漏报
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250522_final_rgb_DA1077_fl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250527_final_rgb_DA1106_szfl_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250527_final_rgb_DA1106_szfl_op_dropoldpairs_ng_cross_pair.csv'),                                        
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250527_final_rgb_DA1106_szfl_op_dropoldpairs_ok_cross_pair.csv'),                                        

                                    # 惠州光弘
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs.csv'),
                                        
                                    # 天津信天冷焊
                                    os.path.join(annotation_folder,
                                                f'aug_train_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs_ok_cross_pair_up_cp5.csv'),
                                    os.path.join(annotation_folder,
                                                f'aug_train_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs_ng_cross_pair_up_cp5.csv'),
                                    os.path.join(annotation_folder,
                                                f'aug_train_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs.csv'),

                                        # trainval_labels_singlepinpad组pair数据
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_nonpaired_rgb.csv'),
                                        # 3D数据
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250423_final_rgb_DA920_3D_dropoldpairs.csv'),

                                        # 广州诺华漏报数据
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250421_final_rgb_DA917_gznh_dropoldpairs.csv'),  

                                        # 辉瑞达，昆山特供处理通用数据
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs_clean3.csv'),  

                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_singlepinpad_merged.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240329_final_model_cleaned.csv'),
                                         # os.path.join(annotation_folder,
                                         #              f'aug_train_pair_labels_{region}_240403debug_final.csv'),
                                         # os.path.join(annotation_folder,
                                         #              f'aug_train_pair_labels_{region}_240404debug_final.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240424_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240428_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240429_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240715_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240702_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240708_final_RGB_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240725_final_rgb_model_cleaned.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_240930_final_rgbv2_update_250630.csv'),
                                         # 3D lighting                                       
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_singlepinpad_241018_final_rgb_update_250630.csv'),

                                                      
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
                                         
                                         ]


        if 'NGonly' in version_name:
            aug_train_pair_data_filenames += [
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_ng_model_cleaned.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240913_final_rgb_update_250630.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240919_final_rgb_ng_update_250630.csv'),
            ]

        else:
            aug_train_pair_data_filenames += [
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_update_250630.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240913_final_rgb_model_cleaned.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240919_final_rgb_model_cleaned.csv'),
            ]

        aug_val_pair_data_filenames = [
                                        #深圳明弘达返回虚焊未检出250825
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_ok_corss_pair_up.csv'),

                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_ng_corss_pair_up.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_ok_corss_pair_up2_250828.csv'),
                                        # # 福建泉州智领返回的虚焊
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_ok_corss_pair_up.csv'),

                                        # 广州华创冷焊未检出
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs_ok_corss_pair_up.csv'),
                                        
                                        # 广州华创虚焊未检出
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs.csv'),
                                        
                                        # 深圳明弘达返回虚焊未检出
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_ok_cross_pair_clean_up.csv'),
                                        
                                        # 江苏昆山丘钛翘脚虚焊漏报
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250707_final_rgb_DA1341_jxks_xh_op_dropoldpairs_clean.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250707_final_rgb_DA1341_jxks_xh_op_dropoldpairs_clean_ng_cross_pair_clean_up.csv'),
                                        # 江苏昆山丘钛人工制作翘脚
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepinpad_250710_final_rgb_DA1388_jsks_xh_op_dropoldpairs.csv'),
                                        # 深圳达人高科翘脚漏报
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250627_final_rgb_DA1292_szdrgk_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250627_final_rgb_DA1292_szdrgk_dropoldpairs_ok_cross_pair_up.csv'),
                                        
                                        # 澜悦翘脚漏报
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250620_final_rgb_DA1241_ly_qj_dropoldpairs.csv'),
                                      
                                    # 永林冷焊
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_up_250522.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up_up_250522.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_ok_cross_pair_up.csv'),
                                        
                                    # 深圳富洛翘脚漏报
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250522_final_rgb_DA1077_fl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250527_final_rgb_DA1106_szfl_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250527_final_rgb_DA1106_szfl_op_dropoldpairs_ok_cross_pair.csv'),                                        
                                        
                                    # 惠州光弘
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs.csv'),
                                        
                                    # 天津信天冷焊
                                    os.path.join(annotation_folder,
                                                f'aug_test_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs_ok_cross_pair_up.csv'),
                                    os.path.join(annotation_folder,
                                                f'aug_test_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs_ng_cross_pair_up.csv'),
                                    os.path.join(annotation_folder,
                                                f'aug_test_pair_labels_singlepinpad_250521_final_rgb_DA1059_tjxt_cold_solder_dropoldpairs.csv'),
                                                                        
                                    # trainval_labels_singlepinpad组pair数据
                                    os.path.join(annotation_folder,
                                                f'aug_val_pair_labels_singlepinpad_nonpairedsplit_rgb_model_cleaned.csv'),
                                    # 3D数据
                                    os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250423_final_rgb_DA920_3D_dropoldpairs.csv'),
                                    # 广州诺华漏报数据
                                    os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250421_final_rgb_DA917_gznh_dropoldpairs_up_250516.csv'), 
                                    # 辉瑞达，昆山特供处理通用数据
                                    os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs_clean3.csv'), 
                                                    
                                    os.path.join(annotation_folder,
                                                     f'aug_val_pair_labels_singlepinpad_merged_update_250630.csv'),
                                        os.path.join(annotation_folder,
                                                     f'aug_val_pair_labels_singlepinpad_240329_final_update_250630.csv'),
                                       os.path.join(annotation_folder, f'aug_test_pair_labels_singlepinpad_240329_final_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_240424_final_RGB_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_240428_final_RGB_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_240429_final_RGB_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_240716_final_RGB_update_250630.csv'),
                                       # 3D lighting                                 
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241018_final_rgb_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241111_final_rgb_D433_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241114_final_rgb_DA472_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241206_final_rgb_DA534_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241206_final_rgb_DA519_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241220_final_rgb_DA1620_dropoldpairs_update_250630.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_241231_final_rgb_DA3031_dropoldpairs_up_250522.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepinpad_250121_final_rgb_DA677_dropoldpairs.csv'),  
                                                
                                       ]
        
        if special_data == 'calibrate_csv':
            aug_train_pair_data_filenames = aug_val_pair_data_filenames
        
        elif special_data == 'filter_train_data':
            aug_train_pair_data_filenames += [
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_update_250630.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240913_final_rgb_model_cleaned.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240919_final_rgb_model_cleaned.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_ng_model_cleaned.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240913_final_rgb_update_250630.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_singlepinpad_240919_final_rgb_ng_update_250630.csv'),
            ]
        
        elif special_data == "hrd_only_bg_jc":
            aug_train_pair_data_filenames = []
            aug_val_pair_data_filenames = []
            annotation_filename = None

            val_annotation_filename = None
            test_annotation_filenames = None


    elif region == 'padgroup':
        annotation_filename = None
        val_annotation_filename = None
        test_annotation_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_{region}_230306v2_rgb_final3.csv')]

        aug_train_pair_data_filenames = [
                                        # 河南郑州卓威电子LED误报连锡
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs.csv'),
                                        
            
                                        # 深圳诚而信LED连锡漏报 (本体包含很多)
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250708_final_rgb_DA1364_cex_led_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250708_final_rgb_DA1364_cex_led_op_dropoldpairs_ok_cross_pair_clean_up.csv'),


                                        # 3D补充连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250707_final_rgb_DA1339_3D_short_dropoldpairs.csv'),
                                        # 深圳名琛连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250707_final_rgb_DA1357_szmc_op_dropoldpairs.csv'),
                                        
                                        # 硬姐连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250703_final_rgb_DA1321_yj_lxop_dropoldpairs_ok_cross_pair_clean.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250703_final_rgb_DA1321_yj_lxop_dropoldpairs.csv'),
                                        # 郑州装联漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl2_dropoldpairs.csv'),
                                        # 江西蓝之洋
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250610_final_rgb_DA1167_1168_jxlzy_2_dropoldpairs.csv'),

                                        # 3D数据
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250423_final_rgb_DA920_3D_dropoldpairs_clean.csv'),


                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_soft_clean.csv'),
                                                                                
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_{region}_v090nonpaired_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                        
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_{region}_v090merged_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_{region}_240407_final_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_selfpair_labels_{region}_240407_final_rgb_final3_model_cleaned_graphite_uncertain.csv'),

                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_240417_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_selfpair_labels_padgroup_240417_final_rgb_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_240418_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_selfpair_labels_padgroup_240418_final_rgb_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_240419_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_selfpair_labels_padgroup_240419_final_rgb_model_cleaned_graphite_uncertain.csv'),

                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_240424_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_240428_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_240429_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                        # 241023     
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_241023_final_rgb_update_241109_model_cleaned_graphite_uncertain.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_241023_final_rgb_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                        # 241101
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_241101_final_rgb_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_padgroup_241101_final_rgb_update_241109_model_cleaned_graphite_uncertain.csv'),

                                        # 2412 DA512, DA513
                                        # os.path.join(annotation_folder,
                                        #               f'aug_train_pair_labels_padgroup_241202_final_rgb_DA512.csv'),                                                      
                                        # os.path.join(annotation_folder,
                                        #               f'aug_train_pair_labels_padgroup_241203_final_rgb_DA513.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_241202_final_rgb_DA512_update_241206_model_cleaned_graphite_uncertain.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_241203_final_rgb_DA513_update_241204_model_cleaned_graphite_uncertain.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_241203_final_rgb_DA513_pad_3_update_241204_model_cleaned_graphite_uncertain.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_241224_final_rgb_DA575and577_drop.csv'),

                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_250115_final_rgb_DA656_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_250118_final_rgb_DA663-670_dropoldpairs.csv'),

                                        # 辉瑞达现场处理的通用数据集，这部分数据集保证了insp引脚干净，但允许ref引脚上有轻微本体，所以ref和insp存在部分不对称数据，
                                        # 因为辉瑞达现场条件较为宽松，可以接受上述情况，在后续模型优化方面要考虑这一点，而且这部分数据全是误报数据，ok量较大的问题也要考虑
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_train_pair_labels_padgroup_250219_final_rgb_DA730_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_train_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs.csv'),

                                        # 第一次粗糙处理
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_train_pair_labels_padgroup_250224_final_rgb_ks_general_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_train_pair_labels_padgroup_250224_final_rgb_hrd_general_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_clean.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_two_pad_short_clean.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_two_pad_clean.csv'),
                                        
                                        # 优化德州漏报，暂时把jsd和仁盈的数据关掉
                                        # # 金赛点误报数据
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_train_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv'),
                                        # # 仁盈误报数据
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_train_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv'),
                                        # 德洲漏报，原始数据做测试集
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_250409_final_rgb_DA857_zz_add_ok_dropoldpairs_cross_pair_ok.csv'), 
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_250409_final_rgb_DA857_zz_add_ok_dropoldpairs_cross_pair_ng.csv'),                                       
                                         ]


        if 'NGonly' in version_name:
            aug_train_pair_data_filenames += [
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_240701_final_RGB_ng_model_cleaned_graphite_uncertain.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_240715_final_RGB_ng_model_cleaned_graphite_uncertain.csv')
            ]

        else:
            aug_train_pair_data_filenames += [
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_240701_final_RGB_up_250707.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_240708_final_RGB_model_cleaned_graphite_uncertain.csv'),

            ]
        

        aug_val_pair_data_filenames = [
                                        # 河南郑州卓威电子LED误报连锡
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs.csv'),
                                        # 深圳诚而信LED连锡漏报 (本体包含很多)
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250708_final_rgb_DA1364_cex_led_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250708_final_rgb_DA1364_cex_led_op_dropoldpairs_ok_cross_pair_clean_up.csv'),

                                        # 3D补充连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250707_final_rgb_DA1339_3D_short_dropoldpairs.csv'),
                                        # 深圳名琛连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250707_final_rgb_DA1357_szmc_op_dropoldpairs.csv'),
                                        
                                        # 硬姐连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250703_final_rgb_DA1321_yj_lxop_dropoldpairs_ok_cross_pair_clean.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250703_final_rgb_DA1321_yj_lxop_dropoldpairs.csv'),
                                        
                                        # 郑州装联漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl2_dropoldpairs.csv'),                                        
                                        # 郑州装联漏报测试集中ng，ok数量较少，扩充一下
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl_dropoldpairs_ng_cross_pair.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_rgb_DA1272_hnzl_dropoldpairs_ok_cross_pair_up.csv'),
                                        
                                        # 江西蓝之洋
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250610_final_rgb_DA1167_1168_jxlzy_2_dropoldpairs.csv'),
                                        # 3D数据
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250423_final_rgb_DA920_3D_dropoldpairs_clean.csv'),


                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_soft_clean.csv'),

                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_250409_final_rgb_DA857_zz_add_ok_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_padgroup_250409_final_rgb_DA857_zz_add_ok_dropoldpairs.csv'),

                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv'),
                                        
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_test_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv'), 
                                       
                                        os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_{region}_v090nonpaired_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_{region}_v090merged_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_{region}_240407_final_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_240407_final_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_padgroup_240417_final_RGB_up_250707.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_padgroup_240418_final_RGB_up_250707.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_padgroup_240419_final_RGB_up_250707.csv'),
                                       
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_240424_final_RGB_up_250707.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_240428_final_RGB_up_250707.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_240429_final_RGB_up_250707.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_padgroup_240715_final_RGB_up_250707.csv'),

                                        # os.path.join(annotation_folder,
                                        #               f'aug_test_pair_labels_padgroup_241023_final_rgb.csv'),

                                        # 241101
                                        os.path.join(annotation_folder,
                                                      f'aug_val_pair_labels_padgroup_241101_final_rgb_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_val_pair_labels_padgroup_241101_final_rgb_update_241109_model_cleaned_graphite_uncertain.csv'),

                                        # 2412 DA512
                                        # os.path.join(annotation_folder,
                                        #               f'aug_test_pair_labels_padgroup_241202_final_rgb_DA512.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_241202_final_rgb_DA512_update_241206_up_250707.csv'),
                                        os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_241224_final_rgb_DA575and577_drop.csv'),  
                                        # 辉瑞达现场处理的通用数据集，这部分数据集保证了insp引脚干净，但允许ref引脚上有轻微本体，所以ref和insp存在部分不对称数据，
                                        # 因为辉瑞达现场条件较为宽松，可以接受上述情况，在后续模型优化方面要考虑这一点                                                    
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_test_pair_labels_padgroup_250219_final_rgb_DA730_dropoldpairs.csv'),   
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_test_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs.csv'), 
                                        #   
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_test_pair_labels_padgroup_250224_final_rgb_ks_general_dropoldpairs.csv'), 
                                        # os.path.join(annotation_folder,
                                        #                 f'aug_test_pair_labels_padgroup_250224_final_rgb_hrd_general_dropoldpairs.csv'),   
                                        os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_clean.csv'),   
                                        os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_two_pad_short_clean.csv'),   
                                        os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_two_pad_clean.csv'),                                                                                 

                                       ]

        if 'masked' in version_name:
            aug_train_pair_data_filenames += [
                                        # 3D补充连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250707_final_rgb_mask_DA1339_3D_short_dropoldpairs.csv'),
                                        # 深圳名琛连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250707_final_rgb_mask_DA1357_szmc_op_dropoldpairs.csv'),
                                        # 硬姐连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250703_final_rgb_mask_DA1321_yj_lxop_dropoldpairs.csv'),
                                        
                                        # 郑州装联漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_rgb_mask_DA1272_hnzl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_rgb_mask_DA1272_hnzl2_dropoldpairs.csv'),
                                        # 3D数据
                                            os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250610_final_rgb_mask_DA1167_1168_jxlzy_2_dropoldpairs.csv'),

                                        # 3D数据
                                            os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250423_final_rgb_mask_DA920_3D_dropoldpairs_clean.csv'),
                                            # os.path.join(annotation_folder,
                                             #  f'aug_train_pair_labels_padgroup_250317_final_rgb_mask_DA786_solder_shortage_fp_dropoldpairs.csv'),
                                                os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_padgroup_240419masked_final_RGB_model_cleaned_graphite_uncertain.csv'),

                                              os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_240424_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                                              os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_240428_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                                              os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_240429_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                                              os.path.join(annotation_folder,
                                                          f'aug_train_pair_labels_padgroup_241023_final_rgb_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                          f'aug_train_pair_labels_padgroup_241224_final_rgb_mask_DA575and577_drop.csv'),
                                        # 辉瑞达现场处理的通用数据集，这部分数据集保证了insp引脚干净，但允许ref引脚上有轻微本体，所以ref和insp存在部分不对称数据，
                                        # 因为辉瑞达现场条件较为宽松，可以接受上述情况，在后续模型优化方面要考虑这一点                                                    
                                            # os.path.join(annotation_folder,
                                            #               f'aug_train_pair_labels_padgroup_250219_final_rgb_mask_DA730_dropoldpairs.csv'),
                                            # os.path.join(annotation_folder,
                                            #               f'aug_train_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs.csv'),  
                                            # os.path.join(annotation_folder,
                                            #               f'aug_train_pair_labels_padgroup_250224_final_rgb_mask_ks_general_dropoldpairs.csv'),                                                      
                                            # os.path.join(annotation_folder,
                                            #               f'aug_train_pair_labels_padgroup_250224_final_rgb_mask_hrd_general_dropoldpairs.csv'),
                                            os.path.join(annotation_folder,
                                                          f'aug_test_pair_labels_padgroup_general_final_white_mask_hrd_ks_dropoldpairs_clean.csv'),                                                      

                                              ]
    
            if 'NGonly' in version_name:
                aug_train_pair_data_filenames += [
                    os.path.join(annotation_folder,
                                 f'aug_train_pair_labels_padgroup_240701_final_RGB_mask_ng_model_cleaned_graphite_uncertain.csv'),
                    os.path.join(annotation_folder,
                                 f'aug_train_pair_labels_padgroup_240715_final_RGB_mask_ng_model_cleaned_graphite_uncertain.csv')
                ]
            else:
                aug_train_pair_data_filenames += [
                    os.path.join(annotation_folder,
                                 f'aug_train_pair_labels_padgroup_240701_final_RGB_mask_up_250707.csv'),
                    os.path.join(annotation_folder,
                                 f'aug_train_pair_labels_padgroup_240708_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                    os.path.join(annotation_folder,
                                 f'aug_train_pair_labels_padgroup_240715_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                ]
            aug_val_pair_data_filenames += [
                                        # 3D补充连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250707_final_rgb_mask_DA1339_3D_short_dropoldpairs.csv'),
                                        # 深圳名琛连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250707_final_rgb_mask_DA1357_szmc_op_dropoldpairs.csv'),
                                        # 硬姐连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250703_final_rgb_mask_DA1321_yj_lxop_dropoldpairs.csv'),
                                      
                                        # 郑州装联漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250625_final_rgb_mask_DA1272_hnzl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250625_final_rgb_mask_DA1272_hnzl2_dropoldpairs.csv'),
                                        # 3D数据
                                            os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250610_final_rgb_mask_DA1167_1168_jxlzy_2_dropoldpairs.csv'),

                                            # 3D数据
                                            os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250423_final_rgb_mask_DA920_3D_dropoldpairs_clean.csv'),
                                            # os.path.join(annotation_folder,
                                    #               f'aug_test_pair_labels_padgroup_250317_final_rgb_mask_DA786_solder_shortage_fp_dropoldpairs.csv'), 
                                            os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_padgroup_240419masked_final_RGB_up_250707.csv'),
                                            os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_240419masked_final_RGB_up_250707.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_padgroup_240424_final_RGB_mask_up_250707.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_padgroup_240428_final_RGB_mask_up_250707.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_padgroup_240429_final_RGB_mask_up_250707.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_240715_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                          f'aug_test_pair_labels_padgroup_241023_final_rgb_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                          f'aug_test_pair_labels_padgroup_241224_final_rgb_mask_DA575and577_drop.csv'),

                                            # 辉瑞达现场处理的通用数据集，这部分数据集保证了insp引脚干净，但允许ref引脚上有轻微本体，所以ref和insp存在部分不对称数据，
                                            # 因为辉瑞达现场条件较为宽松，可以接受上述情况，在后续模型优化方面要考虑这一点
                                            # os.path.join(annotation_folder,
                                            #               f'aug_test_pair_labels_padgroup_250219_final_rgb_mask_DA730_dropoldpairs.csv'),  
                                            # os.path.join(annotation_folder,
                                            #               f'aug_test_pair_labels_padgroup_250219_final_rgb_mask_DA728_filter_clean_dropoldpairs.csv'),  
                                            # os.path.join(annotation_folder,
                                            #               f'aug_test_pair_labels_padgroup_250224_final_rgb_mask_ks_general_dropoldpairs.csv'),                                                                                                                
                                            # os.path.join(annotation_folder,
                                            #               f'aug_test_pair_labels_padgroup_250224_final_rgb_mask_hrd_general_dropoldpairs.csv'), 
                                            os.path.join(annotation_folder,
                                                          f'aug_test_pair_labels_padgroup_general_final_rgb_mask_hrd_ks_dropoldpairs_clean.csv'),                                                                                                                
 
                                            ]


        if 'rgbwhite' in version_name:
            aug_train_pair_data_filenames += [
                                        # 河南郑州卓威电子LED误报连锡
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs.csv'),

                                        # 深圳诚而信LED连锡漏报 (本体包含很多)
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250708_final_white_DA1364_cex_led_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250708_final_white_DA1364_cex_led_op_dropoldpairs_ok_cross_pair_clean_up.csv'),

                                        # 3D补充连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250707_final_white_DA1339_3D_short_dropoldpairs.csv'),
                                        # 深圳名琛连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250707_final_white_DA1357_szmc_op_dropoldpairs.csv'),
                                        # 硬姐连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250703_final_white_DA1321_yj_lxop_dropoldpairs_ok_cross_pair_clean.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250703_final_white_DA1321_yj_lxop_dropoldpairs.csv'),
                                        
                                        # 郑州装联漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_white_DA1272_hnzl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_white_DA1272_hnzl2_dropoldpairs.csv'),
                                            # 江西蓝之洋
                                            os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250610_final_white_DA1167_1168_jxlzy_2_dropoldpairs.csv'),
    
                                            # 3D数据
                                            os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250423_final_white_DA920_3D_dropoldpairs_clean.csv'),
                                            # 德洲漏报，原始数据做测试集
                                            os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_250409_final_white_DA857_zz_add_ok_dropoldpairs_cross_pair_ok.csv'), 
                                            os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_250409_final_white_DA857_zz_add_ok_dropoldpairs_cross_pair_ng.csv'), 
                                            os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_clean.csv'), 
                                            os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_soft_clean.csv'), 
                                                                                        
                                            #  os.path.join(annotation_folder,
                                            #               f'aug_train_pair_labels_padgroup_250403_final_white_DA852_ry_syh_dropoldpairs.csv'),
                                            # os.path.join(annotation_folder,
                                            #                f'aug_train_pair_labels_padgroup_250317_final_white_DA786_solder_shortage_fp_dropoldpairs.csv'),

                                             os.path.join(annotation_folder,
                                                          f'aug_train_pair_labels_{region}_240407_final_white_final3_model_cleaned_graphite_uncertain.csv'),
                                             os.path.join(annotation_folder,
                                                          f'aug_train_selfpair_labels_{region}_240407_final_white_final3_model_cleaned_graphite_uncertain.csv'),
                                             os.path.join(annotation_folder,
                                                          f'aug_train_pair_labels_padgroup_240417_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                             os.path.join(annotation_folder,
                                                          f'aug_train_selfpair_labels_padgroup_240417_final_white_model_cleaned_graphite_uncertain.csv'),
                                             os.path.join(annotation_folder,
                                                          f'aug_train_pair_labels_padgroup_240418_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                             os.path.join(annotation_folder,
                                                          f'aug_train_selfpair_labels_padgroup_240418_final_white_model_cleaned_graphite_uncertain.csv'),
                                             os.path.join(annotation_folder,
                                                          f'aug_train_pair_labels_padgroup_240419_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                             os.path.join(annotation_folder,
                                                          f'aug_train_selfpair_labels_padgroup_240419_final_white_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_240424_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_240428_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_240429_final_WHITE_model_cleaned_graphite_uncertain.csv'),
 
                                            # 2411
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_241023_final_white_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_241023_final_white_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_241101_final_white_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_241101_final_white_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            # 2412
                                            # os.path.join(annotation_folder,
                                            #              f'aug_train_pair_labels_padgroup_241202_final_white_DA512.csv'),
                                            # os.path.join(annotation_folder,
                                            #              f'aug_train_pair_labels_padgroup_241203_final_white_DA513.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_241202_final_white_DA512_update_241206_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_241203_final_white_DA513_update_241204_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_241203_final_white_DA513_pad_3_update_241204_model_cleaned_graphite_uncertain.csv'),

                                            os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_241224_final_white_DA575and577_drop.csv'),

                                            # 辉瑞达现场处理的通用数据集，这部分数据集保证了insp引脚干净，但允许ref引脚上有轻微本体，所以ref和insp存在部分不对称数据，
                                            # 因为辉瑞达现场条件较为宽松，可以接受上述情况，在后续模型优化方面要考虑这一点                                                           
                                            # os.path.join(annotation_folder,
                                            #                f'aug_train_pair_labels_padgroup_250219_final_white_DA730_dropoldpairs.csv'),  
                                            # os.path.join(annotation_folder,
                                            #                f'aug_train_pair_labels_padgroup_250219_final_white_DA728_filter_clean_dropoldpairs.csv'),  
                                            # os.path.join(annotation_folder,
                                            #                f'aug_train_pair_labels_padgroup_250224_final_white_ks_general_dropoldpairs.csv'),                                                                                               
                                            # os.path.join(annotation_folder,
                                            #                f'aug_train_pair_labels_padgroup_250224_final_white_hrd_general_dropoldpairs.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_clean.csv'),                                                                                               
                                            os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_two_pad_short_clean.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_two_pad_clean.csv'),

                                             ]
            aug_val_pair_data_filenames += [
                                        # 河南郑州卓威电子LED误报连锡
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs.csv'),

                                        # 深圳诚而信LED连锡漏报 (本体包含很多)
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250708_final_white_DA1364_cex_led_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250708_final_white_DA1364_cex_led_op_dropoldpairs_ok_cross_pair_clean_up.csv'),

                                        # 3D补充连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250707_final_white_DA1339_3D_short_dropoldpairs.csv'),
                                        # 深圳名琛连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250707_final_white_DA1357_szmc_op_dropoldpairs.csv'),
                                        # 硬姐连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250703_final_white_DA1321_yj_lxop_dropoldpairs_ok_cross_pair_clean.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250703_final_white_DA1321_yj_lxop_dropoldpairs.csv'),
                                        
                                        # 郑州装联漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250625_final_white_DA1272_hnzl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250625_final_white_DA1272_hnzl2_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_white_DA1272_hnzl_dropoldpairs_ng_cross_pair.csv'),

                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_white_DA1272_hnzl_dropoldpairs_ok_cross_pair_up.csv'),
                                            # 江西蓝之洋
                                            os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250610_final_white_DA1167_1168_jxlzy_2_dropoldpairs.csv'),
    
                                            # 3D数据
                                            os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250423_final_white_DA920_3D_dropoldpairs_clean.csv'),
                                           os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_250409_final_white_DA857_zz_add_ok_dropoldpairs.csv'),
                                           os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_250409_final_white_DA857_zz_add_ok_dropoldpairs.csv'),
                                           os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_clean.csv'),
                                           os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_soft_clean.csv'),
                                                                                        
                                            # os.path.join(annotation_folder,
                                            #                f'aug_test_pair_labels_padgroup_250317_final_white_DA786_solder_shortage_fp_dropoldpairs.csv'),
                                        #    os.path.join(annotation_folder,
                                        #                 f'aug_test_pair_labels_padgroup_250403_final_white_DA852_ry_syh_dropoldpairs.csv'),
                                           os.path.join(annotation_folder,
                                                        f'aug_val_pair_labels_{region}_240407_final_white_final3_model_cleaned_graphite_uncertain.csv'),
                                           os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_{region}_240407_final_white_final3_model_cleaned_graphite_uncertain.csv'),
                                           os.path.join(annotation_folder,
                                                        f'aug_val_pair_labels_padgroup_240417_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                           os.path.join(annotation_folder,
                                                        f'aug_val_pair_labels_padgroup_240418_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                           os.path.join(annotation_folder,
                                                        f'aug_val_pair_labels_padgroup_240419_final_WHITE_model_cleaned_graphite_uncertain.csv'),

                                            os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_padgroup_240424_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_padgroup_240428_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_padgroup_240429_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                            # os.path.join(annotation_folder,
                                            #              f'aug_test_pair_labels_padgroup_241023_final_white.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_val_pair_labels_padgroup_241101_final_white_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_val_pair_labels_padgroup_241101_final_white_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_test_pair_labels_padgroup_241202_final_white_DA512_update_241206_model_cleaned_graphite_uncertain.csv'),   
                                             os.path.join(annotation_folder,
                                                           f'aug_test_pair_labels_padgroup_241224_final_white_DA575and577_drop.csv'), 

                                            # 辉瑞达现场处理的通用数据集，这部分数据集保证了insp引脚干净，但允许ref引脚上有轻微本体，所以ref和insp存在部分不对称数据，
                                            # 因为辉瑞达现场条件较为宽松，可以接受上述情况，在后续模型优化方面要考虑这一点                                                            
                                            #  os.path.join(annotation_folder,
                                            #                f'aug_test_pair_labels_padgroup_250219_final_white_DA730_dropoldpairs.csv'),  
                                            #  os.path.join(annotation_folder,
                                            #                f'aug_test_pair_labels_padgroup_250219_final_white_DA728_filter_clean_dropoldpairs.csv'),  
                                            # os.path.join(annotation_folder,
                                            #                f'aug_test_pair_labels_padgroup_250224_final_white_ks_general_dropoldpairs.csv'), 
                                            # os.path.join(annotation_folder,
                                            #                f'aug_test_pair_labels_padgroup_250224_final_white_hrd_general_dropoldpairs.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_clean.csv'), 
                                            os.path.join(annotation_folder,
                                                           f'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_two_pad_short_clean.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_two_pad_clean.csv'),

                                           ]


            if 'masked' in version_name:
                aug_train_pair_data_filenames += [
                                        # 3D补充连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250707_final_white_mask_DA1339_3D_short_dropoldpairs.csv'),
                                        # 深圳名琛连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250707_final_white_mask_DA1357_szmc_op_dropoldpairs.csv'),
                                        # 硬姐连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250703_final_white_mask_DA1321_yj_lxop_dropoldpairs.csv'),
                                        
                                            # 郑州装联漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_white_mask_DA1272_hnzl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250625_final_white_mask_DA1272_hnzl2_dropoldpairs.csv'),
                                            # 江西蓝之洋
                                            os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250610_final_white_mask_DA1167_1168_jxlzy_2_dropoldpairs.csv'),
    
                                                # 3D数据
                                                os.path.join(annotation_folder,
                                                       f'aug_train_pair_labels_padgroup_250423_final_white_mask_DA920_3D_dropoldpairs_clean.csv'),
                                                    # os.path.join(annotation_folder,
                                                    #           f'aug_train_pair_labels_padgroup_250317_final_white_mask_DA786_solder_shortage_fp_dropoldpairs.csv'),  
                                                    os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_240419masked_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                                  os.path.join(annotation_folder,
                                                               f'aug_train_pair_labels_padgroup_240424_final_WHITE_mask_model_cleaned_graphite_uncertain.csv'),
                                                  os.path.join(annotation_folder,
                                                               f'aug_train_pair_labels_padgroup_240428_final_WHITE_mask_model_cleaned_graphite_uncertain.csv'),
                                                  os.path.join(annotation_folder,
                                                               f'aug_train_pair_labels_padgroup_240429_final_WHITE_mask_model_cleaned_graphite_uncertain.csv'),
                                                  os.path.join(annotation_folder,
                                                              f'aug_train_pair_labels_padgroup_241023_final_white_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                                  os.path.join(annotation_folder,
                                                              f'aug_train_pair_labels_padgroup_241224_final_white_mask_DA575and577_drop.csv'),
                                                  os.path.join(annotation_folder,
                                                              f'aug_train_pair_labels_padgroup_250219_final_white_mask_DA730_dropoldpairs.csv'),
                                                os.path.join(annotation_folder,
                                                              f'aug_train_pair_labels_padgroup_250219_final_white_mask_DA728_filter_clean_dropoldpairs.csv'),
                                                # os.path.join(annotation_folder,
                                                #               f'aug_train_pair_labels_padgroup_250224_final_white_mask_ks_general_dropoldpairs.csv'),
                                                                                                            
                                                #  os.path.join(annotation_folder,
                                                #               f'aug_train_pair_labels_padgroup_250224_final_white_mask_hrd_general_dropoldpairs.csv'),   
                                                os.path.join(annotation_folder,
                                                              f'aug_train_pair_labels_padgroup_general_final_white_mask_hrd_ks_dropoldpairs_clean.csv'),
                                                                                                            
                                                            
                                                  ]
                aug_val_pair_data_filenames += [
                                        # 3D补充连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250707_final_white_mask_DA1339_3D_short_dropoldpairs.csv'),
                                        # 深圳名琛连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250707_final_white_mask_DA1357_szmc_op_dropoldpairs.csv'),
                                        
                                        # 硬姐连锡漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250703_final_white_mask_DA1321_yj_lxop_dropoldpairs.csv'),
                                        
                                        # 郑州装联漏报
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250625_final_white_mask_DA1272_hnzl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250625_final_white_mask_DA1272_hnzl2_dropoldpairs.csv'),
                                            # 江西蓝之洋
                                            os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250610_final_white_mask_DA1167_1168_jxlzy_2_dropoldpairs.csv'),
    
                                                # 3D数据
                                                os.path.join(annotation_folder,
                                                       f'aug_test_pair_labels_padgroup_250423_final_white_mask_DA920_3D_dropoldpairs_clean.csv'),
                                                # os.path.join(annotation_folder,
                                                #               f'aug_test_pair_labels_padgroup_250317_final_white_mask_DA786_solder_shortage_fp_dropoldpairs.csv'),  
                                                os.path.join(annotation_folder,
                                                        f'aug_val_pair_labels_padgroup_240419masked_final_WHITE_model_cleaned_graphite_uncertain.csv'),
                                                os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_240419masked_final_WHITE_model_cleaned_graphite_uncertain.csv'),

                                                os.path.join(annotation_folder,
                                                             f'aug_test_pair_labels_padgroup_240424_final_WHITE_mask_model_cleaned_graphite_uncertain.csv'),
                                                os.path.join(annotation_folder,
                                                             f'aug_test_pair_labels_padgroup_240428_final_WHITE_mask_model_cleaned_graphite_uncertain.csv'),
                                                os.path.join(annotation_folder,
                                                             f'aug_test_pair_labels_padgroup_240429_final_WHITE_mask_model_cleaned_graphite_uncertain.csv'),
                                                os.path.join(annotation_folder,
                                                              f'aug_test_pair_labels_padgroup_241023_final_white_mask_update_241109_model_cleaned_graphite_uncertain.csv'),
                                                os.path.join(annotation_folder,
                                                              f'aug_test_pair_labels_padgroup_241224_final_white_mask_DA575and577_drop.csv'), 

                                                # 辉瑞达现场处理的通用数据集，这部分数据集保证了insp引脚干净，但允许ref引脚上有轻微本体，所以ref和insp存在部分不对称数据，
                                                # 因为辉瑞达现场条件较为宽松，可以接受上述情况，在后续模型优化方面要考虑这一点                                                              
                                                # os.path.join(annotation_folder,
                                                #               f'aug_test_pair_labels_padgroup_250219_final_white_mask_DA730_dropoldpairs.csv'),  
                                                # os.path.join(annotation_folder,
                                                #               f'aug_test_pair_labels_padgroup_250219_final_white_mask_DA728_filter_clean_dropoldpairs.csv'), 
                                                # os.path.join(annotation_folder,
                                                #               f'aug_test_pair_labels_padgroup_250224_final_white_mask_ks_general_dropoldpairs.csv'), 
                                                # os.path.join(annotation_folder,
                                                #               f'aug_test_pair_labels_padgroup_250224_final_white_mask_hrd_general_dropoldpairs.csv'),   
                                                os.path.join(annotation_folder,
                                                              f'aug_test_pair_labels_padgroup_general_final_white_mask_hrd_ks_dropoldpairs_clean.csv'), 
                                                                                                                                                     
                                                ]

        if special_data == 'calibrate_csv':
            aug_train_pair_data_filenames = aug_val_pair_data_filenames
        elif special_data == 'test_csv':
            test_annotation_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_{region}_230306v2_rgb_final3.csv')]
            aug_train_pair_data_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_{region}_230306v2_rgb_final3.csv')]
            aug_val_pair_data_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_{region}_230306v2_rgb_final3.csv')]
        elif special_data == "down_all_csv":
            pass
        elif special_data == "ceshi":
            aug_train_pair_data_filenames = [
                # os.path.join(annotation_folder,
                #                 f'aug_train_pair_labels_padgroup_250708_final_rgb_DA1364_cex_led_op_dropoldpairs.csv'),
                # os.path.join(annotation_folder,
                #                 f'aug_train_pair_labels_padgroup_250708_final_white_DA1364_cex_led_op_dropoldpairs.csv'),


                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv'),

                os.path.join(annotation_folder,
                                f'aug_train_pair_labels_padgroup_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                os.path.join(annotation_folder,
                                f'aug_test_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                os.path.join(annotation_folder,
                                f'aug_test_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv'),
                os.path.join(annotation_folder,
                                f'aug_test_pair_labels_padgroup_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv'),
                                ]
            aug_val_pair_data_filenames = aug_train_pair_data_filenames
            # test_annotation_filenames = None

    elif region == 'singlepad':
        annotation_filename = None
        val_annotation_filename = None
        test_annotation_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_singlepad_230306v2_model_cleaned2_update_250508.csv')]
        aug_train_pair_data_filenames = [
                                        # 深圳朗特短引脚翘脚
                                        # os.path.join(annotation_folder,
                                        #             f'defect_labels_singlepad_rgb_pairs_split_train_ok_cross_pair_up_clean.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'defect_labels_singlepad_rgb_pairs_split_train_ng_cross_pair_up_clean.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'defect_labels_singlepad_rgb_pairs_split_train_ng_cross_pair_up_clean2_up.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'defect_labels_singlepad_rgb_pairs_split_train_ok_cross_pair_up_clean2_up.csv'),
                                        
                                        # 广州华创冷焊漏报 #TODO 该部分虚焊数据很重要，需要重新整理一下
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs_ok_corss_pair_up.csv'),
                                                                        
                                        # 深圳硬姐虚焊少锡漏报（3D数据）
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs_ok_cross_pair_clean_up.csv'),
                                        
                                        # 南京菲林&泰克尔曼虚焊漏报
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_ok_cross_pair_clean_up.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_clean.csv'),
                                        
                                        # 永林冷焊
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_cross_ng_diff_material_up.csv'),

                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_ok_cross_pair_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs.csv'),
                                           
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_cross_ng_diff_material_up.csv'),
                                           
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_cross_ng_diff_material_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv'),
                                           
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_ng_cross_pair.csv'),
                                        
                                        # 惠州光弘
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs.csv'),
                                        
                                        # 3D数据
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250423_final_rgb_DA920_3D_dropoldpairs_clean.csv'),
                                        
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
                                                     f'aug_train_pair_labels_singlepad_240507_final_RGB_model_cleaned2_update_250508.csv'),
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
                                        # 广州华创冷焊漏报（3D数据）
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepad_250812_final_rgb_DA1574_gzhc_lh_dropoldpairs_ok_corss_pair_up.csv'),
                                                                 
                                        # 深圳朗特短引脚翘脚
                                        # os.path.join(annotation_folder,
                                        #             f'defect_labels_singlepad_rgb_pairs_split_train_ok_cross_pair_up_for_test.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'defect_labels_singlepad_rgb_pairs_split_test_ok_cross_pair_up_clean.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'defect_labels_singlepad_rgb_pairs_split_test_ng_cross_pair_up_clean.csv'),

                                        # 深圳硬姐虚焊少锡漏报（3D数据）
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs_ok_cross_pair_clean_up.csv'),
                                        # 南京菲林&泰克尔曼虚焊漏报
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_ok_cross_pair_clean_up.csv'),
                                        # os.path.join(annotation_folder,
                                        #             f'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_clean.csv'),

                                        # 永林冷焊
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_cross_ng_diff_material_up.csv'),

                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_ok_cross_pair_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_cross_ng_diff_material2_up.csv'),
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv'),
                                        
                                        # 惠州光弘
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs.csv'),
                                        
                                        # 3D数据
                                        os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_singlepad_nonpaired_rgb_model_cleaned.csv'),
                                        # 3D数据
                                        os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_singlepad_250423_final_rgb_DA920_3D_dropoldpairs_clean.csv'),
                                        os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs.csv'),
                                        os.path.join(annotation_folder,f'aug_val_pair_labels_singlepad_merged_model_cleaned2_update_250508.csv'),
                                        os.path.join(annotation_folder, f'aug_val_pair_labels_singlepad_240329_final_model_cleaned2.csv'),
                                        os.path.join(annotation_folder, f'aug_test_pair_labels_singlepad_240329_final_update_250516.csv'),                                                                        
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_240428_final_RGB_update_250516.csv'),
                                                                        
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_240429_final_RGB_update_250516.csv'),
                                       os.path.join(annotation_folder,f'aug_val_pair_labels_singlepad_240808_final_RGB_model_cleaned2_update_250508.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241018_final_rgb_update_250516.csv'),
                                       # 3D lighting  
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241103_final_rgb_update_250516.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241111_final_rgb_D433_update_250516.csv'),
                                    #    os.path.join(annotation_folder,
                                    #                   f'aug_test_pair_labels_singlepad_241114_final_rgb_DA465.csv'),
                                    
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241114_final_rgb_DA472_update_250516.csv'),

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241126_final_rgb_DA505_update_250516.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241126_final_rgb_DA507_update_250516.csv'),

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241206_final_rgb_DA534_update_250512.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_241206_final_rgb_DA519_update_250512.csv'),    

                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs_update_250512.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250110_final_rgb_DA629-2_dropoldpairs_update_250512.csv'),
                                       os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_general_final_rgb_hrd_ks_dropoldpairs_clean2.csv'), 
                                       ]

        if special_data == "test_csv":
            test_annotation_filenames = [os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs.csv'),]
            aug_train_pair_data_filenames = [
                                        # 金赛点误报筛选后数据
                                        os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs.csv'),            
                                        ]
            aug_val_pair_data_filenames = [
                                        os.path.join(annotation_folder,f'aug_test_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs.csv'),
                                        ]   
        elif special_data == 'calibrate_csv':
            aug_train_pair_data_filenames = aug_val_pair_data_filenames
            aug_train_pair_data_filenames = [
                                        # 金赛点误报筛选后数据
                                        os.path.join(annotation_folder,
                                                    f'all_black_sample_results.csv'),            
                                        ]
            
            # aug_val_pair_data_filenames = aug_val_pair_data_filenames
            # test_annotation_filenames = test_annotation_filenames

    elif region == 'body':
        annotation_filename = None
        val_annotation_filename = None
        # test_annotation_filenames = None
        # annotation_filename = os.path.join(annotation_folder, f'aug_train_pair_labels_body_nonpaired_resorted_model_cleaned_250123_update6_250418.csv')
        # val_annotation_filename = os.path.join(annotation_folder, f'aug_val_pair_labels_body_nonpaired_resorted_model_cleaned_250123_update6_250418.csv')
        test_annotation_filenames = os.path.join(annotation_folder, f'bbtest_labels_pairs_body_nonpaired_resorted_model_cleaned_250123_update6_250418.csv')
        aug_train_pair_data_filenames = [
                                        #  广东深圳兰顺现场返回chip件损坏
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250725_final_white_DA1472_gzls_chip_op_dropoldpairs.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250725_final_white_DA1472_gzls_chip_op_dropoldpairs_ok_cross_pair_clean.csv'),

                                        # 东莞紫檀山替代料错件
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_cp3.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_cp3_transpose.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair_cp5.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair_cp5_transpose.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ok_cross_pair_cp5.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ok_insp_cross_pair_cp5.csv'),

                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_insp_ok_cross_pair_up.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair_transpose.csv'),
                                            os.path.join(annotation_folder, f'aug_train_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_ok_cross_pair.csv'),                                                                                      
                                        # 上海仟铂损件漏报
                                          os.path.join(annotation_folder, f'aug_train_pair_labels_body_250521_final_white_DA1062_shqb_dropoldpairs.csv'),
                                          
                                        # 高测损件漏报
                                          os.path.join(annotation_folder, f'aug_train_pair_labels_body_250520_final_white_DA1057_gc_op2_dropoldpairs_ok_cross_pair.csv'),
                                          os.path.join(annotation_folder, f'aug_train_pair_labels_body_250520_final_white_DA1057_gc_op2_dropoldpairs.csv'),
                                        
                                        
                                        os.path.join(annotation_folder, f'aug_train_pair_labels_body_nonpaired_resorted_model_cleaned_250123_update6_250418.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_body_250407_final_white_DA855_jlc_dropoldpairs_update6_250418.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_body_250407_final_white_DA855_jlc_dropoldpairs_cross_pair_ok_update6_250418.csv'),
                                        # cross pair用于训练，原始数据用于测试

                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_body_mergedresorted_model_cleaned_250306_update6_250418.csv'),
                                         os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_body_240328b_finalresorted_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                                        f'aug_train_selfpair_labels_body_240328b_finalresorted_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_body_merged_withcpmlresorted_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_body_240725_final_white_model_cleaned_250306_update6_250418.csv'),
#                                             os.path.join(annotation_folder,
#                                                          f'aug_train_pair_labels_{region}_240913_final_white.csv'),
                                            os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_body_240918_final_white_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_body_240919_final_white_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_body_240926sub_final_white_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_body_241023_final_white_20573_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_body_241023_final_white_20583_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_body_241023_final_white_240071_model_cleaned_250306_update6_250418.csv'),

                                            # LED cross pair 241030
                                        os.path.join(annotation_folder,
                                        f'LED_cross_pairs_ok_train_model_cleaned_250306_update6_250418.csv'),
                                        os.path.join(annotation_folder,
                                        f'LED_cross_pairs_ng_train_model_cleaned_250306_update6_250418.csv'),                                         

                                        #  melf cross pair 241031
                                        os.path.join(annotation_folder,
                                        f'body_241031_melf_ng_cross_pair2_train_model_cleaned_250306_update6_250418.csv'),
                                        os.path.join(annotation_folder,
                                        f'body_241031_melf_ok_cross_pair2_train_model_cleaned_250306_update6_250418.csv'),
                                        os.path.join(annotation_folder,
                                        f'aug_train_pair_labels_body_241118_final_white_model_cleaned_250306_update6_250418.csv'),
     
                                        # os.path.join(annotation_folder,
                                        # f'body_white_paris_final_train_refine2.csv'),
                                        #  os.path.join(annotation_folder,
                                        # f'body_rgb_paris_final_train_refine2.csv'),

                                        os.path.join(annotation_folder,
                                        f'aug_train_pair_labels_body_250122_final_white_DA680_dropoldpairs_model_cleaned_250306_update6_250418.csv'),
                                        os.path.join(annotation_folder,
                                        f'corss_ok_pair_DA680_model_cleaned_250306_update6_250418.csv'),
                                        os.path.join(annotation_folder,
                                        f'aug_corss_ng_pair_DA680_model_cleaned_250306_update6_250418.csv'),
                                        os.path.join(annotation_folder,
                                        f'aug_train_pair_labels_body_250122_final_white_DA682_dropoldpairs_update6_250418.csv'),   
                                        # DA727高测翻转漏报数据，原始数据只有3个缺陷jpair，增强后对train上采样到5，原始train加入到test
                                        # os.path.join(annotation_folder,
                                        #  f'aug_train_pair_labels_body_250217_final_white_DA727_dropoldpairs_up.csv'),  
                                        os.path.join(annotation_folder,
                                            f'aug_train_pair_labels_body_250217_final_white_DA727_dropoldpairs_model_cleaned_250306_update6_250418.csv'), 
                                        os.path.join(annotation_folder,
                                            f'aug_train_pair_labels_body_250218_final_white_DA727_dropoldpairs_ok_cross_pair_update6_250418.csv'),  
                                        os.path.join(annotation_folder,
                                            f'aug_train_pair_labels_body_250218_final_white_DA727_dropoldpairs_ng_cross_pair_update6_250418.csv'), 
                                            # 按照不同料号组合的错件
                                        os.path.join(annotation_folder,
                                            f'defect_labels_body_white_pairs_aug_ng_test_soft_bad_size2_update6_250418.csv'), 
                                        os.path.join(annotation_folder,
                                                         f'body_white_paris_final_test_refine2_model_cleaned_250306_update6_250418.csv'),
                                        os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_body_250526_final_white_DA88888_2d_led_dropoldpairs.csv'),                                
            ]
        if 'newonly' not in version_name:
                aug_train_pair_data_filenames.append(os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_body_240913_final_white_model_cleaned_250306.csv'))
                                                            
        aug_val_pair_data_filenames = [
                                        #  广东深圳兰顺现场返回chip件损坏
                                            os.path.join(annotation_folder, f'aug_test_pair_labels_body_250725_final_white_DA1472_gzls_chip_op_dropoldpairs.csv'),
                                            os.path.join(annotation_folder, f'aug_test_pair_labels_body_250725_final_white_DA1472_gzls_chip_op_dropoldpairs_ng_cross_pair_clean.csv'),
                                            os.path.join(annotation_folder, f'aug_test_pair_labels_body_250725_final_white_DA1472_gzls_chip_op_dropoldpairs_ok_cross_pair_clean.csv'),

                                        os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_body_250526_final_white_DA88888_2d_led_dropoldpairs.csv'),  
                                # 东莞紫檀山替代料错件
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_transpose.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair_transpose.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ok_cross_pair.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250522_final_white_DA1079_dwzts_led_dropoldpairs_ok_insp_cross_pair.csv'),
                            
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_insp_ok_cross_pair_up.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair_up.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_ng_cross_pair_up_transpose.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250526_final_white_DA1079_dwzts_led_dropoldpairs_ok_cross_pair.csv'),
                                                                       
                                    # 上海仟铂损件漏报
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250521_final_white_DA1062_shqb_dropoldpairs.csv'),
                                    # 高科润损件漏报

                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250520_final_white_DA1057_gc_op2_dropoldpairs.csv'),

                                    os.path.join(annotation_folder, f'aug_val_pair_labels_body_nonpaired_resorted_model_cleaned_250123_update6_250418.csv'),
                                    os.path.join(annotation_folder, f'aug_test_pair_labels_body_250407_final_white_DA855_jlc_dropoldpairs_update6_250418.csv'),

                                    os.path.join(annotation_folder, f'aug_val_pair_labels_body_mergedresorted_model_cleaned_250306_update6_250418.csv'),
                                                                               
                                            os.path.join(annotation_folder, f'aug_val_pair_labels_body_240328b_finalresorted_model_cleaned_250407_update6_250418.csv'),
                                            os.path.join(annotation_folder, f'aug_test_pair_labels_body_240328b_finalresorted_model_cleaned_250306_2_update6_250418.csv'),
                                                                                                  
                                           os.path.join(annotation_folder,
                                                      f'aug_val_pair_labels_body_merged_withcpmlresorted_model_cleaned_250306_update6_250418.csv'),
                                           os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_body_240918sub_final_white_model_cleaned_250306_update6_250418.csv'),
                                           os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_body_240919sub_final_white_model_cleaned_250306_update6_250418.csv'),
                                           os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_body_241023_final_white_20573_model_cleaned_250306_update6_250418.csv'),
                                           os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_body_241023_final_white_20583_model_cleaned_250306_update6_250418.csv'),
#             os.path.join(annotation_folder, f'aug_test_pair_labels_{region}_240819_final_white.csv')
                                              # LED cross pair 241030
                                            os.path.join(annotation_folder,
                                            f'LED_cross_pairs_ng_val_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                            f'LED_cross_pairs_ok_val_model_cleaned_250306_update6_250418.csv'),

                                            # melf cross pair 241031
                                            os.path.join(annotation_folder,
                                            f'body_241031_melf_ng_cross_pair2_val_model_cleaned_250306_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                            f'body_241031_melf_ok_cross_pair2_val_model_cleaned_250306_update6_250418.csv'),
                                            
                                            os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_body_241118_final_white_model_cleaned_250306_update6_250418.csv'),

                                            # os.path.join(annotation_folder,
                                            # f'body_white_paris_final_test_refine2.csv'),
                                            #  os.path.join(annotation_folder,
                                              # f'body_rgb_paris_final_test_refine2.csv'),
                                            os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_body_250122_final_white_DA680_dropoldpairs_model_cleaned_250306_2_update6_250418.csv'), 
                                                
                                            os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_body_250122_final_white_DA682_dropoldpairs_model_cleaned_250306_2_update6_250418.csv'),
                                            # DA727高测翻转漏报数据，原始数据只有3个缺陷jpair，增强后对train上采样到5，原始train加入到test   
                                            os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_body_250217_final_white_DA727_dropoldpairs_model_cleaned_250306_2_update6_250418.csv'),
                                              
                                            os.path.join(annotation_folder,
                                            f'aug_test_pair_labels_body_250218_final_white_DA727_dropoldpairs_model_cleaned_250306_2_update6_250418.csv'),
                                                 
                                            os.path.join(annotation_folder,
                                            f'aug_train_pair_labels_body_250218_final_white_DA727_dropoldpairs_model_cleaned_250306_update6_250418.csv'),
                                              

                                            # 按照不同料号组合的错件，经过测试，下面两个文件用于训练
                                            os.path.join(annotation_folder,
                                             f'defect_labels_body_white_pairs_aug_ng_train_size_match2_update6_250418.csv'),
                                            os.path.join(annotation_folder,
                                             f'defect_labels_body_white_pairs_aug_ng_train_hard_bad_size2_update6_250418.csv'),  
                                             os.path.join(annotation_folder,
                                                         f'body_white_paris_final_train_refine2_model_cleaned_250306_update6_250418.csv'),
                                           ]
        # # if 'bdg' in version_name:
        #         aug_train_pair_data_filenames.append(os.path.join(annotation_folder,
        #                                                  f'body_white_paris_final_test_refine2_model_cleaned_250306_update6_250418.csv'))
        #         aug_val_pair_data_filenames.append(os.path.join(annotation_folder,
        #                                                  f'body_white_paris_final_train_refine2_model_cleaned_250306_update6_250418.csv'))
        if special_data == 'test_csv':
            aug_train_pair_data_filenames = [test_annotation_filenames]
            aug_val_pair_data_filenames = aug_train_pair_data_filenames

    return annotation_filename, val_annotation_filename, test_annotation_filenames, aug_train_pair_data_filenames, aug_val_pair_data_filenames
    