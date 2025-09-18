import os

special_csvs_name = {
    'body':{
        # 鸿通宇漏报  
        # 'dwzts_supply':{
        #     'train': ['aug_train_pair_labels_body_250526_final_white_DA88888_2d_led_dropoldpairs.csv'],

        #     'val': ['aug_test_pair_labels_body_250526_final_white_DA88888_2d_led_dropoldpairs.csv',],       
        # },
        # 鸿通宇漏报  
        'hty_op':{
            'train': ['aug_train_pair_labels_body_250319_final_white_DA796_797_dropoldpairs_all_cross_pair_ng.csv',
                      'aug_train_pair_labels_body_250319_final_white_DA796_797_dropoldpairs_all_cross_pair_ng_up2.csv'],

            'val': ['aug_test_pair_labels_body_250319_final_white_DA796_797_dropoldpairs.csv',
                    'aug_train_pair_labels_body_250319_final_white_DA796_797_dropoldpairs.csv',
                    'aug_train_pair_labels_body_250319_final_white_DA796_797_dropoldpairs_tg2.csv',],       
        },

        # 鸿通宇漏报ok数据
        '796797':{
            'train': ['aug_train_pair_labels_body_250319_final_white_DA796_797_dropoldpairs_all_cross_pair_ok_bad.csv'],
            'val': [],       
        },
    },
    'singlepad': {

        # 深圳迅航
        'szxh': {
            'train': ['aug_train_pair_labels_singlepad_250728_final_rgb_DA1486_szxh_xh_op_dropoldpairs_ok_corss_pair.csv',
                        'aug_train_pair_labels_singlepad_250728_final_rgb_DA1486_szxh_xh_op_dropoldpairs.csv'
                      ],

            'val': ['aug_test_pair_labels_singlepad_250728_final_rgb_DA1486_szxh_xh_op_dropoldpairs_ok_corss_pair_up.csv',
                    'aug_test_pair_labels_singlepad_250728_final_rgb_DA1486_szxh_xh_op_dropoldpairs_ng_corss_pair.csv',
                    'aug_test_pair_labels_singlepad_250728_final_rgb_DA1486_szxh_xh_op_dropoldpairs.csv',
                    ],
        },


        # 深圳朗特FAE
        'szlt_FAE': {
            'train': ['defect_labels_singlepad_rgb_pairs_split_train_ok_cross_pair_up.csv',
                        'defect_labels_singlepad_rgb_pairs_split_train_ng_cross_pair_up.csv'
                      ],

            'val': ['defect_labels_singlepad_rgb_pairs_split_test_ok_cross_pair_up.csv',
                    'defect_labels_singlepad_rgb_pairs_split_test_ng_cross_pair_up.csv',
                    ],
        },

        'szlt_clean': {
            'train': ['defect_labels_singlepad_rgb_pairs_split_train_ok_cross_pair_up_clean.csv',
                    'defect_labels_singlepad_rgb_pairs_split_train_ng_cross_pair_up_clean.csv',
                    'defect_labels_singlepad_rgb_pairs_split_train_ok_cross_pair_up_clean2_up.csv',
                    'defect_labels_singlepad_rgb_pairs_split_train_ng_cross_pair_up_clean2_up.csv',

                      ],

            'val': [
                    'defect_labels_singlepad_rgb_pairs_split_train_ok_cross_pair_up_for_test.csv',
                    'defect_labels_singlepad_rgb_pairs_split_test_ok_cross_pair_up_clean.csv',
                    'defect_labels_singlepad_rgb_pairs_split_test_ng_cross_pair_up_clean.csv',

                    ],
        },
        'szlt_TG': {
            'train': ['defect_labels_singlepad_rgb_pairs_split_train_ok_cross_pair_up_TG.csv',
                      ],

            'val': ['defect_labels_singlepad_rgb_pairs_split_test_ok_cross_pair_up_TG.csv',
                    ],
        },


        # 重庆国讯暗色漏报，特供
        'cqgx':{  
            'train': ['aug_train_pair_labels_singlepad_250701_final_rgb_DA1306_cqgx_dropoldpairs.csv',

                      ],
                    
            'val': [
                    'aug_test_pair_labels_singlepad_250701_final_rgb_DA1306_cqgx_dropoldpairs.csv',

                    ],  
                },

        # 深圳硬姐虚焊漏报TG
        'szyj_clean':{  
            'train': ['aug_train_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs_TG.csv',

                      ],
                    
            'val': [
                    'aug_test_pair_labels_singlepad_250630_final_rgb_DA1301_yj_op_dropoldpairs_TG.csv',

                    ],  
                },
        # 南京菲林&泰克尔曼clean
        'njfl_tkr_clean':{  
            'train': ['aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_ok_cross_pair_clean_up.csv',
                      'aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_clean.csv',

                      ],
                    
            'val': [
                    'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_ok_cross_pair_clean_up.csv',
                    'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_clean.csv',

                    ],  
                      },
        # 惠州光弘误报特供
        # 南京菲林&泰克尔曼ok特供
        'njfl_tkr':{  
            'train': ['aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_TG_ok_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_TG.csv',

                      ],
                    
            'val': [
                    'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_TG_ok_cross_pair_up.csv',
                    'aug_test_pair_labels_singlepad_250625_final_rgb_DA1271_njfl_tkr_dropoldpairs_TG.csv',

                    ],  
                      },
        # 惠州光弘误报特供
        'hzgh_soft':{  
            'train': ['aug_train_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_soft_TG_ok_cross_pair_up.csv',

                      ],
                    
            'val': [
                    'aug_test_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_soft_TG_ok_cross_pair_up.csv',

                    ],  
                      },
        'hzgh_hard':{  
            'train': ['aug_train_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_hard_TG_ok_cross_pair_up.csv',

                      ],
                    
            'val': [
                    'aug_test_pair_labels_singlepad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_hard_TG_ok_cross_pair_up.csv',

                    ],  
                      },
        # 炉前炉后
        'bp_stove_clean':{  
            'train': ['aug_test_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_ok_cross_pair_up.csv',

                      ],
                    
            'val': [
                    'aug_train_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_ok_cross_pair_up.csv',

                    ],  
                      },
        # 炉前炉后
        'bp_stove_soft':{  
            'train': ['aug_test_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft.csv',
                      'aug_train_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft_ok_cross_pair.csv',

                      ],
                    
            'val': [
                    'aug_train_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft.csv',
                    'aug_test_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft_ok_cross_pair.csv',

                    ],  
                      },

        # 炉前炉后
        'bp_stove_TG':{  
            'train': ['aug_test_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_TG.csv',
                      'aug_train_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_TG_ok_cross_pair_up.csv',

                      ],
                    
            'val': [
                    'aug_train_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_TG.csv',
                    'aug_test_pair_labels_singlepad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_TG_ok_cross_pair_up.csv',

                    ],  
                      },

        # 永林冷焊
        'yl_lh':{  
            'train': ['aug_train_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_cross_ng_diff_material_up.csv',
                      'aug_train_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_ok_cross_pair_up.csv',

                      'aug_train_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_cross_ng_diff_material_up.csv',
                      # 部分少锡仅在test中，所以增强部分test做训练
                      'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_cross_ng_diff_material_up.csv',

                      'aug_train_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_ng_cross_pair.csv',


                      ],
            
            'val': ['aug_test_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_cross_ng_diff_material_up.csv',
                    'aug_test_pair_labels_singlepad_250508_final_rgb_DA985_yl_dropoldpairs_ok_cross_pair_up.csv',

                    'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up.csv',
                    'aug_test_pair_labels_singlepad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_cross_ng_diff_material2_up.csv',
                    'aug_test_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv',
                    ],       
        },
        # 永林冷焊
        'yl_lh_TG':{  
            'train': [
                    'aug_test_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_TG.csv',
                      ],
            'val': [
                    'aug_train_pair_labels_singlepad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_TG.csv',
                    ],       
        },

        'jsd':{  # 错位导致金板，检测板的框不对齐
            'train': ['aug_train_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs_TG.csv'],
            'val': ['aug_test_pair_labels_singlepad_250318_final_rgb_DA794_jsd_fp_dropoldpairs_TG.csv'],       
        },
        'SMTAOITS-1441': {
            'train': ['aug_train_pair_labels_singlepad_250103_final_rgb_DA620_dropoldpairs.csv'],
            'val': ['aug_test_pair_labels_singlepad_250103_final_rgb_DA620_dropoldpairs.csv'],
        },
        'DA703': {
            'train': ['aug_train_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs.csv'],
            'val': ['aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs.csv'],
        },
        'DA703-cross_pair': {
            'train': ['aug_train_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv'],
            'val': ['aug_test_pair_labels_singlepad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv'],
        },
        # 测评结果,DA703的特供模型在该份数据集上表现也不好
        'DA712_250214': {
            'train': ['aug_train_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs_filter.csv',
                      'aug_train_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv'],
            'val': ['aug_test_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs_filter.csv', 
                    'aug_test_pair_labels_singlepad_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv'],
        },
    },

    'singlepinpad': {
       # #深圳明弘达返回虚焊未检出250825特供
        'szmhd_250825_TG': {
            'train': [  
                        'aug_train_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_TG.csv',
                        'aug_train_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_TG_ok_corss_pair_up.csv',

                      
                      ],

            'val': [
                        'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_TG.csv',
                        'aug_test_pair_labels_singlepinpad_250825_final_rgb_DA1644_szmhd_xh_op_dropoldpairs_TG_ok_corss_pair_up.csv',

                    ],
        },  
       # 福建泉州智领返回的ok特供
        'fzqzzl_clean': {
            'train': [  
                        'aug_test_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs.csv',
                        'aug_train_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_ok_corss_pair_up_250825.csv',
                        'aug_train_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_ok_corss_pair2_up_250825.csv',
                        'aug_test_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_ok_corss_pair2_up_250825.csv',
                      
                      ],

            'val': [
                        'aug_train_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs.csv',
                        'aug_test_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_ok_corss_pair_up_250825.csv',

                    ],
        },                                             

       # 福建泉州智领返回的ok特供
        'fzqzzl_TG': {
            'train': [  
                        'aug_train_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_TG_ok_corss_pair.csv',
                        'aug_train_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_TG.csv',
                      
                      ],

            'val': [
                        'aug_test_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_TG_ok_corss_pair.csv',
                        'aug_test_pair_labels_singlepinpad_250819_final_rgb_DA1619_fzqzzl_xh_op_dropoldpairs_TG.csv',

                    ],
        }, 
        # 深圳钦盛伟源原始数据
        'szqswy_ori': {
            'train': [  
                        'defect_labels_singlepinpad_rgb_pairs_split_train.csv',
                      
                      ],

            'val': [
                        'defect_labels_singlepinpad_rgb_pairs_split_test.csv',
                    ],
        }, 
        # 深圳钦盛伟源原始数据
        'szqswy_aug': {
            'train': [  
                        'aug_train_pair_labels_singlepinpad_250814_final_rgb_DA1587_szqswy_dropoldpairs.csv',
                      
                      ],

            'val': [
                        'aug_test_pair_labels_singlepinpad_250814_final_rgb_DA1587_szqswy_dropoldpairs.csv',
                    ],
        }, 


        # 广州华创ok特供
        'gzhc_TG': {
            'train': [  
                        'aug_train_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs_TG_ok_corss_pair.csv',
                        'aug_train_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs_TG.csv',
                      
                      ],

            'val': [
                        'aug_test_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs_TG_ok_corss_pair.csv',
                        'aug_test_pair_labels_singlepinpad_250806_final_rgb_DA1551_gzhc_xh_op_dropoldpairs_TG.csv',
                    ],
        },     
        # 南京焊兆虚焊漏报
        'njzh': {
            'train': [  
                        'aug_train_pair_labels_singlepinpad_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv',
                        'aug_train_pair_labels_singlepinpad_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv',
                      
                      ],

            'val': [
                        'aug_test_pair_labels_singlepinpad_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv',
                        'aug_test_pair_labels_singlepinpad_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv',
                    ],
        },                     
        
        # 卓威误报特供
        'szzw_clean': {
            'train': ['aug_train_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_clean.csv',
                        'aug_train_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_ok_corss_pair_up.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_clean.csv',
                    'aug_test_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_ok_corss_pair_up.csv'
                    ],
        },

        # 卓威误报特供
        'szzw_clean': {
            'train': ['aug_train_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_clean.csv',
                        'aug_train_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_ok_corss_pair_up.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_clean.csv',
                    'aug_test_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs_ok_corss_pair_up.csv'
                    ],
        },
        'szzw_TG': {
            'train': ['aug_test_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs.csv',
                      ],

            'val': ['aug_train_pair_labels_singlepinpad_250728_final_rgb_DASMT2098_dropoldpairs.csv'
                    ],
        },

        # 南京淼龙虚焊漏报
        'njml_clean': {
            'train': ['aug_train_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_clean_ok_corss_pair_up.csv',
                        'aug_train_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_clean.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_clean_ok_corss_pair_up.csv',
                    'aug_test_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_clean.csv'
                    ],
        },

        # OK特供
        'njml_ok_TG': {
            'train': ['aug_train_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ok_TG.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ok_TG.csv'
                    ],
        },

        # ng特供
        'njml_ng_soft': {
            'train': ['aug_train_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ng_soft.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ng_soft.csv'
                    ],
        },

        # ng特供
        'njml_ng_hard': {
            'train': ['aug_train_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ng_hard.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250728_final_rgb_DA1493_njml_xh_op_dropoldpairs_ng_hard.csv'
                    ],
        },

        # 深圳明弘达返回虚焊未检出(纯TG)
        'szmh_xh_op_onlyNG': {
            'train': ['aug_train_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_TG_ng_cross_pair_clean.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_TG_ng_cross_pair_clean.csv',
                    ],
        },

        'szmh_xh_op_TG': {
            'train': ['aug_train_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_TG_ng_cross_pair_clean.csv',
                      'aug_train_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_TG.csv'
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_TG_ng_cross_pair_clean.csv',
                    'aug_test_pair_labels_singlepinpad_250721_final_rgb_DA1461_szmh_xh_op_dropoldpairs_TG.csv'
                    ],
        },


        # 辉瑞达新旧数据整理增强版（斌哥，老板）
        'hrd_DA614532': {
            'train': ['aug_train_pair_labels_singlepinpad_250708_final_rgb_DA1356_hrd614532_op_dropoldpairs.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250708_final_rgb_DA1356_hrd614532_op_dropoldpairs.csv',
                    ],
        },
        'hrd_DA614532_ori': {
            'train': ['defect_labels_singlepinpad_rgb_pairs_split_hrd_da614532_ori_train.csv',
                      ],

            'val': ['defect_labels_singlepinpad_rgb_pairs_split_hrd_da614532_ori_test.csv',
                    ],
        },

        # 江苏昆山丘钛翘脚虚焊漏报特供
        'jsksqt_TG': {
            'train': ['aug_train_pair_labels_singlepinpad_250707_final_rgb_DA1341_jxks_xh_op_dropoldpairs_TG.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250707_final_rgb_DA1341_jxks_xh_op_dropoldpairs_TG.csv',
                    ],
        },

        # 江苏昆山丘钛人工制作翘脚
        'jsksqt_qj_made_by_human': {
            'train': ['aug_train_pair_labels_singlepinpad_250710_final_rgb_DA1388_jsks_xh_op_dropoldpairs.csv',
                        'aug_train_pair_labels_singlepinpad_250710_final_rgb_DA1388_jsks_xh_op_dropoldpairs_ng_cross_pair_clean_up.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250710_final_rgb_DA1388_jsks_xh_op_dropoldpairs.csv',
                    ],
        },

        # 辉瑞达新旧数据整理增强版（斌哥，老板）
        'bg_jc_aug': {
            'train': ['aug_train_pair_labels_singlepinpad_250701_final_rgb_hrd_DA_ALL_dropoldpairs_bg_jc_aug.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250701_final_rgb_hrd_DA_ALL_dropoldpairs_bg_jc_aug.csv',
                    ],
        },
        # 辉瑞达新旧数据整理原始版本，ok降采样（斌哥，老板）
        'bg_jc_ori': {
            'train': [ 'defect_labels_singlepinpad_rgb_pairs_split_jc_bg_ori_train_add_ng_up.csv',

                      ],

            'val': ['defect_labels_singlepinpad_rgb_pairs_split_jc_bg_ori_test_sample_ng_up.csv',
                    ],
        },
        # 辉瑞达旧数据二次整理（jm,robin）
        'hrd_jm_robin': {
            'train': ['aug_train_pair_labels_singlepinpad_250701_final_rgb_DA_ALL_OLD_dropoldpairs_jm_robin.csv',
                    'aug_train_pair_labels_singlepinpad_250703_final_rgb_DA1310_hrd_supply_dropoldpairs.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250701_final_rgb_DA_ALL_OLD_dropoldpairs_jm_robin.csv',
                    'aug_test_pair_labels_singlepinpad_250703_final_rgb_DA1310_hrd_supply_dropoldpairs.csv',
                    ],
        },

        # 深圳达人高科翘脚漏报ok特供
        'szdrgk': {
            'train': ['aug_train_pair_labels_singlepinpad_250627_final_rgb_DA1292_szdrgk_dropoldpairs_TG.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250627_final_rgb_DA1292_szdrgk_dropoldpairs_TG.csv',
                    ],
        },
        # 辉瑞达LED翘脚虚焊漏报，机械开关偏移虚焊漏报（与一般的虚焊成像不同，仅作为特供）
        'hrd_led_switch_op': {
            'train': ['aug_train_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj2_ng_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj2_ng_dropoldpairs_ng_cross_pair.csv',
                      'aug_train_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj2_ng_dropoldpairs_ok_cross_pair_up.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj2_ng_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj2_ng_dropoldpairs_ng_cross_pair.csv',
                    'aug_test_pair_labels_singlepinpad_250618_final_rgb_DA1224_hrd_qj2_ok_dropoldpairs_ng_cross_pair.csv',
                    ],
        },
        'hrd_switch_fp': {
            'train': ['aug_train_pair_labels_singlepinpad_250619_final_rgb_DA1232_hrd_switch_fp_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250619_final_rgb_DA1232_hrd_switch_fp_dropoldpairs_ok_cross_pair_up.csv'
],

            'val': ['aug_test_pair_labels_singlepinpad_250619_final_rgb_DA1232_hrd_switch_fp_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250619_final_rgb_DA1232_hrd_switch_fp_dropoldpairs_ok_cross_pair_up.csv'
],
        },

        # 江西蓝之洋误报漏报特供数据（ok和缺陷很像）
        'jxlzy_aug': {
            'train': ['aug_train_pair_labels_singlepinpad_250612_final_rgb_DA1183_jxlzy_solder_dropoldpairs.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250612_final_rgb_DA1183_jxlzy_solder_dropoldpairs.csv',
                    
                    ],
        },
        'jxlzy_ori': {
            'train': ['aug_train_pair_labels_singlepinpad_250612_final_rgb_DA1183_jxlzy_solder_dropoldpairs_ori.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250612_final_rgb_DA1183_jxlzy_solder_dropoldpairs_ori.csv',
                    
                    ],
        },

        'jxlzy_ori_merge': {
            'train': ['aug_train_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori_update_merge_ng.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori_update_merge_ng.csv',
                    
                    ],
        },

        'jxlzy_ori_base': {
            'train': ['aug_train_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori.csv',
                    
                    ],
        },

        'jxlzy_ori_update': {
            'train': ['aug_train_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori_update.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250617_final_rgb_DA1183_jxlzy_confirm_ng_dropoldpairs_ori_update.csv',
                    
                    ],
        },

        # 深圳利速达特供
        'szlsd_ori':{  
            'train': ['aug_train_pair_labels_singlepinpad_250605_final_rgb_DA1138_szlsd_dropoldpairs_ori.csv'],
                    
            'val': [
                    'aug_test_pair_labels_singlepinpad_250605_final_rgb_DA1138_szlsd_dropoldpairs_ori.csv',
                    ],  
                      },
        'szlsd':{  
            'train': ['aug_train_pair_labels_singlepinpad_250605_final_rgb_DA1138_szlsd_dropoldpairs.csv',

                      ],
                    
            'val': [
                    'aug_test_pair_labels_singlepinpad_250605_final_rgb_DA1138_szlsd_dropoldpairs.csv'],  
                      },
        # 惠州光弘误报特供
        'hzgh_soft':{  
            'train': ['aug_train_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_soft_TG_ok_cross_pair_up.csv'],
                    
            'val': ['aug_test_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_soft_TG_ok_cross_pair_up.csv'],

                      },
        'hzgh_hard':{  
            'train': ['aug_train_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_hard_TG_ok_cross_pair_up.csv',

                      ],
                    
            'val': [
                    'aug_test_pair_labels_singlepinpad_250526_final_rgb_DA1104_hzgh_fp_dropoldpairs_hard_TG_ok_cross_pair_up.csv',

                    ],  
                      },
        # 炉前炉后
        'bp_stove_clean':{  
            'train': ['aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_ok_cross_pair_ok_cross_pair_up.csv',

                      ],
                    
            'val': [
                    'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_ok_cross_pair_ok_cross_pair_up.csv',

                    ],  
                      },

        # 炉前炉后
        'bp_stove_soft':{  
            'train': ['aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft.csv',
                      'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft_ok_cross_pair_up.csv',

                      ],
                    
            'val': [
                    'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft.csv',
                    'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1002_jc_cold_solder_dropoldpairs_soft_ok_cross_pair_up.csv',

                    ],  
                      },

        'yl_lh':{  
            'train': [
                    'aug_train_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs.csv',
                    'aug_train_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up.csv',
                    # test数据更丰富一点
                    'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv',
                    'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_ok_cross_pair_up.csv',

                      ],
            
            'val': [
                    'aug_test_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_up_250522.csv',
                    'aug_test_pair_labels_singlepinpad_250509_final_rgb_DA997_yl_cold_solder_dropoldpairs_ok_cross_pair_up_up_250522.csv',

                    'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_ok_cross_pair_up.csv'

                    ],       
        },
        'yl_lh_TG':{  
            'train': [
                        'aug_test_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_TG.csv',

                      ],
            
            'val': [
                    'aug_train_pair_labels_singlepinpad_250512_final_rgb_DA1007_yl_cold_solder2_dropoldpairs_TG.csv',

                    ],       
        },
        'gz_nh_op':{
            'train':['aug_train_pair_labels_singlepinpad_250421_final_rgb_DA917_gznh_dropoldpairs.csv'],
            'val': ['aug_test_pair_labels_singlepinpad_250421_final_rgb_DA917_gznh_dropoldpairs.csv'],
        },
        'hrd_ks_general_clean':{
            'train':['aug_train_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs_clean3.csv'],
            'val': ['aug_test_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs_clean3.csv'],
        },
        'hrd_ks_general_jm':{
            'train':['aug_train_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs.csv'],
            'val': ['aug_test_pair_labels_singlepinpad_general_final_rgb_hrd_ks_dropoldpairs.csv'],
        },

        'DA730745_aug_ng': {
            'train': ['aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_supply.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv',

                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',

                      
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv',
                    
                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',
                    ],
        },

        'DA730_add_DA745_aug_ng': {
            'train': ['aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_supply.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv',

                    'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv',

                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',
                    ],
        },
        'DA730FAE_add_DA745_aug_ng': {
            'train': ['aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv',
                      
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',                      
                      ],
                      
            'val': ['aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv',
                    'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv',
                    
                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',                    
                    ],
        },
        'DA730_add_DA745': {
            'train': ['aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_supply.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv',
                      
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',                      
                      
                      ],

            'val': ['aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv',
                    
                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',
                    ],
        },
        'DA730FAE_add_DA745': {
            'train': ['aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_30_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA730_28_dropoldpairs_cross_pair_up.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA745_28_dropoldpairs_cross_pair_up.csv',
                      
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                      'aug_train_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',
                      ],
                      
            'val': ['aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv',
                    'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_aug_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250224_final_rgb_DA745_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA745_30_dropoldpairs_cross_pair_up.csv',
                    
                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ng.csv',
                    'aug_test_pair_labels_singlepinpad_250225_final_rgb_DA747_dropoldpairs_cross_pair_ok_up.csv',
                    ],
        },

   #################################################################
        'DA730FAE': {
            'train': ['aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv',
                      'aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'],
                      
            'val': ['aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_processed_by_FAE.csv',
                    'aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv'],
        },

        'DA730': {
            'train': ['aug_train_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs_supply.csv',
                      'aug_train_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'],

            'val': ['aug_test_pair_labels_singlepinpad_250218_final_rgb_DA728_dropoldpairs.csv',
                    'aug_test_pair_labels_singlepinpad_250219_final_rgb_DA730_dropoldpairs.csv'],
        },

        'DA703': {
            'train': ['aug_train_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs.csv'],
            'val': ['aug_test_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs.csv'],
        },
        'DA703-cross_pair': {
            'train': ['aug_train_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv'],
            'val': ['aug_test_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv'],
        },
        # 测评结果,DA703的特供模型在该份数据集上表现也不好
        'DA712_250213': {
            'train': ['aug_train_pair_labels_singlepinpad_250213_final_rgb_DA712_dropoldpairs_cross_pair.csv'],
            'val': ['aug_test_pair_labels_singlepinpad_250213_final_rgb_DA712_dropoldpairs.csv'],
        },
        'DA712_250214': {
            'train': ['aug_train_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs_filter.csv',
                      'aug_train_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv'],
            'val': ['aug_test_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs_filter.csv', 
                    'aug_test_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv'],
        },
    },
    'padgroup': {

        #南京焊兆连锡漏保(仅白图缺陷特征明显, rgb的ng很不明显)
        'njzh_rgbwhite': {
            'train': ['aug_train_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv',
                      'aug_train_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv',
                      'aug_train_pair_labels_padgroup_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv',
                    #   'aug_train_pair_labels_padgroup_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv'
                      ],

            'val': ['aug_test_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv',
                    'aug_test_pair_labels_padgroup_250804_final_white_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv',
                    # 'aug_test_pair_labels_padgroup_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ng_corss_pair_up.csv',
                    'aug_test_pair_labels_padgroup_250804_final_rgb_DA1524_lx_xh_op_dropoldpairs_ok_corss_pair_up.csv'
                    ],
        },
        #  河南郑州卓威电子LED误报连锡
        'hnzzzw_rgb': {
            'train': ['aug_train_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs.csv',
                      ],

            'val': ['aug_test_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs.csv',
                    ],
        },
        'hnzzzw_rgb_TG': {
            'train': ['aug_train_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs_TG.csv',
                      ],

            'val': ['aug_test_pair_labels_padgroup_250717_final_rgb_DA1434_zzzw_lx_op_dropoldpairs_TG.csv',
                    ],
        },

        'hnzzzw_white': {
            'train': ['aug_train_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs.csv',
                      ],

            'val': ['aug_test_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs.csv',
                    ],
        },
        'hnzzzw_white_TG': {
            'train': ['aug_train_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs_TG.csv',
                      ],

            'val': ['aug_test_pair_labels_padgroup_250717_final_white_DA1434_zzzw_lx_op_dropoldpairs_TG.csv',
                    ],
        },


        # 深圳长城白图漏报（rgb图太不明显,将白图加入rgb图）,暂时当做特供数据处理，待训练后看效果
        'szcc_lx_op': {
            'train': ['aug_train_pair_labels_padgroup_250711_final_rgb_DA1397_szcc_lx_op_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250711_final_rgb_mask_DA1397_szcc_lx_op_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250711_final_white_DA1397_szcc_lx_op_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250711_final_white_mask_DA1397_szcc_lx_op_dropoldpairs.csv',
                      ],

            'val': ['aug_test_pair_labels_padgroup_250711_final_rgb_DA1397_szcc_lx_op_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250711_final_rgb_mask_DA1397_szcc_lx_op_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250711_final_white_DA1397_szcc_lx_op_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250711_final_white_mask_DA1397_szcc_lx_op_dropoldpairs.csv',
                    ],
        },
        
        # 韶关嘉立创3D细微漏报
        'jxlzy_TG': {
            'train': ['aug_train_pair_labels_padgroup_250610_final_rgb_DA1167_1168_jxlzy_2_dropoldpairs_TG.csv',
                      'aug_train_pair_labels_padgroup_250610_final_white_DA1167_1168_jxlzy_2_dropoldpairs_TG.csv',
                      ],

            'val': ['aug_test_pair_labels_padgroup_250610_final_rgb_DA1167_1168_jxlzy_2_dropoldpairs_TG.csv',
                    'aug_test_pair_labels_padgroup_250610_final_white_DA1167_1168_jxlzy_2_dropoldpairs_TG.csv',
                    
                    ],
        },


        # 韶关嘉立创3D细微漏报
        'sgjlc_op_rgb': {
            'train': ['aug_train_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs_ok_cross_pair_up.csv',
                      'aug_train_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs.csv',                      
                      ],

            'val': ['aug_test_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs_ng_cross_pair_up.csv',
                    'aug_test_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs_ok_cross_pair_up.csv',
                    
                    ],
        },
        'sgjlc_op_rgbwhite': {
            'train': ['aug_train_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs_ok_cross_pair_up.csv',
                      'aug_train_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs.csv', 

                      'aug_train_pair_labels_padgroup_250422_final_white_DA922_sgjlc2_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250422_final_white_DA922_sgjlc2_dropoldpairs_ok_cross_pair_up.csv',

                      ],

            'val': ['aug_test_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs_ng_cross_pair_up.csv',
                    'aug_test_pair_labels_padgroup_250422_final_rgb_DA922_sgjlc2_dropoldpairs_ok_cross_pair_up.csv',
                    
                    'aug_test_pair_labels_padgroup_250422_final_white_DA922_sgjlc2_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250422_final_white_DA922_sgjlc2_dropoldpairs_ng_cross_pair_up.csv',
                    'aug_test_pair_labels_padgroup_250422_final_white_DA922_sgjlc2_dropoldpairs_ok_cross_pair_up.csv',
                    
                    ],
        },

        # 信润漏报
        'xr_op_rgb': {
            'train': ['aug_train_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs_ng_cross_pairs_up.csv',
                      'aug_train_pair_labels_padgroup_250411_final_rgb_mask_DA873_szxr_dropoldpairs.csv',
                      
                      ],

            'val': ['aug_test_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs_ng_cross_pairs_up.csv',
                    'aug_test_pair_labels_padgroup_250411_final_rgb_mask_DA873_szxr_dropoldpairs.csv',
                    
                    ],
        }, 
        'xr_op_rgbwhite': {
            'train': ['aug_train_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs_ng_cross_pairs_up.csv',
                      'aug_train_pair_labels_padgroup_250411_final_rgb_mask_DA873_szxr_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250411_final_white_DA873_szxr_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250411_final_white_DA873_szxr_dropoldpairs_ng_cross_pair_up.csv',
                      'aug_train_pair_labels_padgroup_250411_final_white_mask_DA873_szxr_dropoldpairs.csv',
                      
                      ],

            'val': ['aug_test_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250411_final_rgb_DA873_szxr_dropoldpairs_ng_cross_pairs_up.csv',
                    'aug_test_pair_labels_padgroup_250411_final_rgb_mask_DA873_szxr_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250411_final_white_DA873_szxr_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250411_final_white_DA873_szxr_dropoldpairs_ng_cross_pair_up.csv',
                    'aug_test_pair_labels_padgroup_250411_final_white_mask_DA873_szxr_dropoldpairs.csv',
                    ],
        },  

        'ry_jsd_rgb': {
            'train': ['aug_train_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250317_final_rgb_mask_DA786_solder_shortage_fp_dropoldpairs.csv',
                      
                      ],

            'val': ['aug_test_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250317_final_rgb_mask_DA786_solder_shortage_fp_dropoldpairs.csv'
                    
                    ],
        },  

        'ry_jsd_rgbwhite': {
            'train': ['aug_train_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250317_final_rgb_mask_DA786_solder_shortage_fp_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250403_final_white_DA852_ry_syh_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250317_final_white_DA786_solder_shortage_fp_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250317_final_white_mask_DA786_solder_shortage_fp_dropoldpairs.csv'
                      ],

            'val': ['aug_test_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250317_final_rgb_mask_DA786_solder_shortage_fp_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250317_final_white_DA786_solder_shortage_fp_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250403_final_white_DA852_ry_syh_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250317_final_white_mask_DA786_solder_shortage_fp_dropoldpairs.csv'
                    ],
        },      

        'smooth_clean_rgb': {
            'train': ['aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv'],

            'val': ['aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv'],
        },
        'smooth_clean_rgbwhite': {
            'train': ['aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv',
                      'aug_train_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_clean.csv'],

            'val': ['aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv',
                    'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_clean.csv'],
        },
        'smooth_soft_clean_rgb': {
            'train': ['aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv',
                      'aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_soft_clean.csv',],

            'val': ['aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv',
                    'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_soft_clean.csv',],
        },
        # 'smooth_soft_clean_rgbwhite': { 已加入常规训练
        #     'train': ['aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv',
        #               'aug_train_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_clean.csv',

        #               'aug_train_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_soft_clean.csv',
        #               'aug_train_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_soft_clean.csv'],

        #     'val': ['aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_clean.csv',
        #             'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_clean.csv',

        #             'aug_test_pair_labels_padgroup_general_final_rgb_hrd_ks_dropoldpairs_smooth_soft_clean.csv',
        #             'aug_test_pair_labels_padgroup_general_final_white_hrd_ks_dropoldpairs_smooth_soft_clean.csv',
        #             ],
    

        # },

        # 石墨，soft_hard_bad数据最好都只添加rgb图，因为这些数据的rgb图和真实ng的rgb图是有区别的，但是白图，尤其是soft_hard_bad的白图和真实ng的难以区分
        # 而且这些数据都不添加mask数据
        'ks_graphite_ok': {
            'train': ['aug_train_pair_labels_padgroup_250224_final_rgb_ks_general_dropoldpairs_graphite_ok.csv'],

            'val': ['aug_test_pair_labels_padgroup_250224_final_rgb_ks_general_dropoldpairs_graphite_ok.csv'],
        },

        'hrd_soft_bad': {
            'train': ['aug_train_pair_labels_padgroup_250224_final_rgb_hrd_general_dropoldpairs_soft.csv'],

            'val': ['aug_test_pair_labels_padgroup_250224_final_rgb_hrd_general_dropoldpairs_soft.csv'],
        },
        'hrd_hard_bad': {
            'train': ['aug_train_pair_labels_padgroup_250224_final_rgb_hrd_general_dropoldpairs_hard.csv'],

            'val': ['aug_test_pair_labels_padgroup_250224_final_rgb_hrd_general_dropoldpairs_hard.csv'],
        },
        'hrd_extrm_hard_bad': {
            'train': ['aug_train_pair_labels_padgroup_250224_final_rgb_hrd_general_dropoldpairs_extrem_hard.csv'],

            'val': ['aug_test_pair_labels_padgroup_250224_final_rgb_hrd_general_dropoldpairs_extrem_hard.csv'],
        },

        'DA728softrgb': {
            'train': ['aug_train_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs_soft_clean.csv'],

            'val': ['aug_test_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs_soft_clean.csv'],
        },
        'DA728softwhite': {
            'train': ['aug_train_pair_labels_padgroup_250219_final_white_DA728_filter_clean_dropoldpairs_soft_clean.csv'],

            'val': ['aug_test_pair_labels_padgroup_250219_final_white_DA728_filter_clean_dropoldpairs_soft_clean.csv'],
        },
        'DA728hardrgb': {
            'train': ['aug_train_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs_add_all_bad.csv'],

            'val': ['aug_test_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs_add_all_bad.csv'],
        },
        'DA728hardwhite': {
            'train': ['aug_train_pair_labels_padgroup_250219_final_white_DA728_filter_clean_dropoldpairs_add_all_bad.csv'],

            'val': ['aug_test_pair_labels_padgroup_250219_final_white_DA728_filter_clean_dropoldpairs_add_all_bad.csv'],
        },
        
        'DA703rgb': {
            'train': ['aug_train_pair_labels_padgroup_250213_final_rgb_DA703_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_rgb_mask_DA703_dropoldpairs.csv'],

            'val': ['aug_test_pair_labels_padgroup_250213_final_rgb_DA703_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_rgb_mask_DA703_dropoldpairs.csv'],
        },
        'DA703rgbwhite': {
            'train': ['aug_train_pair_labels_padgroup_250213_final_rgb_DA703_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_rgb_mask_DA703_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_white_DA703_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_white_mask_DA703_dropoldpairs.csv'],

            'val': ['aug_test_pair_labels_padgroup_250213_final_rgb_DA703_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_rgb_mask_DA703_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_white_DA703_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_white_mask_DA703_dropoldpairs.csv'],
        },
        'DA703rgbhard': {
            'train': ['aug_train_pair_labels_padgroup_250213_final_rgb_DA703_hardbad_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_rgb_mask_DA703_hardbad_dropoldpairs.csv'],

            'val': ['aug_test_pair_labels_padgroup_250213_final_rgb_DA703_hardbad_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_rgb_mask_DA703_hardbad_dropoldpairs.csv'],
        },
        'DA703rgbwhitehard': {
            'train': ['aug_train_pair_labels_padgroup_250213_final_rgb_DA703_hardbad_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_rgb_mask_DA703_hardbad_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_white_DA703_hardbad_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_white_mask_DA703_hardbad_dropoldpairs.csv'],

            'val': ['aug_test_pair_labels_padgroup_250213_final_rgb_DA703_hardbad_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_rgb_mask_DA703_hardbad_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_white_DA703_hardbad_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_white_mask_DA703_hardbad_dropoldpairs.csv'],
        },

        'DA712rgb': {
            'train': ['aug_train_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_rgb_mask_DA712_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs_cross_pair.csv',

                      'aug_train_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv',
                      'aug_train_pair_labels_padgroup_250214_final_rgb_mask_DA712_dropoldpairs.csv',

                      ],

            'val': ['aug_test_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_rgb_mask_DA712_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs_cross_pair.csv',

                    'aug_test_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv',
                    'aug_test_pair_labels_padgroup_250214_final_rgb_mask_DA712_dropoldpairs.csv',
                    ],
        },
        
        'DA712rgbwhite': {
            'train': ['aug_train_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_rgb_mask_DA712_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs_cross_pair.csv',

                      'aug_train_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv',
                      'aug_train_pair_labels_padgroup_250214_final_rgb_mask_DA712_dropoldpairs.csv',

                      'aug_train_pair_labels_padgroup_250213_final_white_DA712_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250213_final_white_DA712_dropoldpairs_cross_pair.csv',
                      'aug_train_pair_labels_padgroup_250213_final_white_mask_DA712_dropoldpairs.csv',
                      
                      'aug_train_pair_labels_padgroup_250214_final_white_DA712_dropoldpairs.csv',
                      'aug_train_pair_labels_padgroup_250214_final_white_DA712_dropoldpairs_cross_pair.csv',
                      'aug_train_pair_labels_padgroup_250214_final_white_mask_DA712_dropoldpairs.csv',
                      ],

            'val': ['aug_test_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_rgb_mask_DA712_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs_cross_pair.csv',

                    'aug_test_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs_cross_pair.csv',
                    'aug_test_pair_labels_padgroup_250214_final_rgb_mask_DA712_dropoldpairs.csv',

                    'aug_test_pair_labels_padgroup_250213_final_white_DA712_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250213_final_white_DA712_dropoldpairs_cross_pair.csv',
                    'aug_test_pair_labels_padgroup_250213_final_white_mask_DA712_dropoldpairs.csv',

                    'aug_test_pair_labels_padgroup_250214_final_white_DA712_dropoldpairs.csv',
                    'aug_test_pair_labels_padgroup_250214_final_white_DA712_dropoldpairs_cross_pair.csv',
                    'aug_test_pair_labels_padgroup_250214_final_white_mask_DA712_dropoldpairs.csv',

                    ],
        },

    }
}


# 特供测试集
        # # DA712昆山启佳特供
        # val_image_pair_path_rgb_ks1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs.csv')
        # val_image_pair_path_rgb_ks2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250214_final_rgb_DA712_dropoldpairs.csv')
        # val_image_pair_path_rgb_ks3 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs.csv')
        # val_image_pair_path_rgb_ks4 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250213_final_rgb_DA712_dropoldpairs.csv')
        
        # val_image_pair_path_rgb_mask_ks1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250214_final_rgb_mask_DA712_dropoldpairs.csv')
        # val_image_pair_path_rgb_mask_ks2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250214_final_rgb_mask_DA712_dropoldpairs.csv')  
        # val_image_pair_path_rgb_mask_ks3 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250213_final_rgb_mask_DA712_dropoldpairs.csv')
        # val_image_pair_path_rgb_mask_ks4 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250213_final_rgb_mask_DA712_dropoldpairs.csv')  
             
        # val_image_pair_path_white_ks1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250214_final_white_DA712_dropoldpairs.csv')
        # val_image_pair_path_white_ks2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250214_final_white_DA712_dropoldpairs.csv')
        # val_image_pair_path_white_ks3 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250213_final_white_DA712_dropoldpairs.csv')
        # val_image_pair_path_white_ks4 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250213_final_white_DA712_dropoldpairs.csv')
        
        # val_image_pair_path_white_mask_ks1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250214_final_white_mask_DA712_dropoldpairs.csv')
        # val_image_pair_path_white_mask_ks2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250214_final_white_mask_DA712_dropoldpairs.csv')  
        # val_image_pair_path_white_mask_ks3 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250213_final_white_mask_DA712_dropoldpairs.csv')
        # val_image_pair_path_white_mask_ks4 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250213_final_white_mask_DA712_dropoldpairs.csv')
        
        # # DA703昆山启佳特供
        # val_image_pair_path_rgb_ks5 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250213_final_rgb_DA703_dropoldpairs.csv')
        # val_image_pair_path_rgb_ks6 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250213_final_rgb_DA703_dropoldpairs.csv')
        # val_image_pair_path_rgb_hard_ks1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250213_final_rgb_DA703_hardbad_dropoldpairs.csv')
        # val_image_pair_path_rgb_hard_ks2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250213_final_rgb_DA703_hardbad_dropoldpairs.csv')
        
        # val_image_pair_path_white_ks5 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250213_final_white_DA703_dropoldpairs.csv')
        # val_image_pair_path_white_ks6 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250213_final_white_DA703_dropoldpairs.csv')
        
        # val_image_pair_path_white_hard_ks1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250213_final_white_DA703_hardbad_dropoldpairs.csv')
        # val_image_pair_path_white_hard_ks2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250213_final_white_DA703_hardbad_dropoldpairs.csv')


        # # DA703昆山启佳特供数据，仅用于cur_jiraissue
        # val_image_pair_path28 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_test_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs.csv')
        # val_image_pair_path29 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_test_pair_labels_singlepinpad_250210_final_rgb_DA703_dropoldpairs_cross_pair_update.csv')
        
        # val_image_pair_path30 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_test_pair_labels_singlepinpad_250213_final_rgb_DA712_dropoldpairs.csv')
        # val_image_pair_path31 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_train_pair_labels_singlepinpad_250213_final_rgb_DA712_dropoldpairs.csv')
        # val_image_pair_path32 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_test_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs.csv')
        # val_image_pair_path33 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_train_pair_labels_singlepinpad_250214_final_rgb_DA712_dropoldpairs.csv')


        # # 辉瑞达特供   
        # val_image_pair_path_hrd_rgb_general1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_rgb_DA730_dropoldpairs.csv')
        # val_image_pair_path_hrd_rgb_general2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs.csv')                       
        # val_image_pair_path_hrd_rgb_mask_general1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_rgb_mask_DA730_dropoldpairs.csv')
        # val_image_pair_path_hrd_rgb_mask_general2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_rgb_mask_DA728_filter_clean_dropoldpairs.csv') 

        # val_image_pair_path_hrd_rgb_soft1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs_soft_clean.csv')
        # val_image_pair_path_hrd_white_soft1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_white_DA728_filter_clean_dropoldpairs_soft_clean.csv')  

        # val_image_pair_path_hrd_rgb_softhard1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_rgb_DA728_filter_clean_dropoldpairs_add_all_bad.csv')
        # val_image_pair_path_hrd_white_softhard1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_white_DA728_filter_clean_dropoldpairs_add_all_bad.csv')

        # val_image_pair_path_hrd_white_general1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_white_DA730_dropoldpairs.csv')
        # val_image_pair_path_hrd_white_general2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_white_DA728_filter_clean_dropoldpairs.csv')  
        # val_image_pair_path_hrd_white_mask_general1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_white_mask_DA730_dropoldpairs.csv')
        # val_image_pair_path_hrd_white_mask_general2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250219_final_white_mask_DA728_filter_clean_dropoldpairs.csv')       

        # # 辉瑞达补充测试集,黑盒测试集
        # val_image_pair_path_hrd_rgb_supply_clean1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250220_final_rgb_DA730_supply_dropoldpairs.csv')
        # val_image_pair_path_hrd_rgb_supply_clean2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250220_final_rgb_DA730_supply_dropoldpairs.csv')  
        # val_image_pair_path_hrd_white_supply_clean1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250220_final_white_DA730_supply_dropoldpairs.csv')
        # val_image_pair_path_hrd_white_supply_clean2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250220_final_white_DA730_supply_dropoldpairs.csv')
        
        # val_image_pair_path_hrd_rgb_supply_soft1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250220_final_rgb_DA730_supply_dropoldpairs_soft.csv')
        # val_image_pair_path_hrd_rgb_supply_soft2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250220_final_rgb_DA730_supply_dropoldpairs_soft.csv')  
        # val_image_pair_path_hrd_white_supply_soft1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250220_final_white_DA730_supply_dropoldpairs_soft.csv')
        # val_image_pair_path_hrd_white_supply_soft2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250220_final_white_DA730_supply_dropoldpairs_soft.csv')  

        # val_image_pair_path_hrd_rgb_supply_hard1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250220_final_rgb_DA730_supply_dropoldpairs_hard.csv')
        # val_image_pair_path_hrd_rgb_supply_hard2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250220_final_rgb_DA730_supply_dropoldpairs_hard.csv')  
        # val_image_pair_path_hrd_white_supply_hard1 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_test_pair_labels_padgroup_250220_final_white_DA730_supply_dropoldpairs_hard.csv')
        # val_image_pair_path_hrd_white_supply_hard2 = os.path.join(image_folder, 'merged_annotation', date,
        #                             f'aug_train_pair_labels_padgroup_250220_final_white_DA730_supply_dropoldpairs_hard.csv') 
        
        # val_image_pair_hrd_black_test747_path1 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_test_pair_labels_padgroup_250225_final_rgb_DA747_dropoldpairs.csv')
        # val_image_pair_hrd_black_test747_path2 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_train_pair_labels_padgroup_250225_final_rgb_DA747_dropoldpairs.csv')
        # val_image_pair_hrd_black_test747_path3 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_test_pair_labels_padgroup_250225_final_rgb_DA747_dropoldpairs_soft.csv')
        # val_image_pair_hrd_black_test747_path4 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_train_pair_labels_padgroup_250225_final_rgb_DA747_dropoldpairs_soft.csv')
        # val_image_pair_hrd_black_test747_path5 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_test_pair_labels_padgroup_250225_final_rgb_DA747_dropoldpairs_hard.csv')
        # val_image_pair_hrd_black_test747_path6 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_train_pair_labels_padgroup_250225_final_rgb_DA747_dropoldpairs_hard.csv')

        # # 测试下面两个文件夹的时候要把padgroup_slide_DA747_250225_just_test换成padgroup_slide_DA747_250225，而把padgroup_slide_DA747_250225另外保存
        # val_image_pair_hrd_black_test747_path7 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_test_pair_labels_padgroup_250225_final_rgb_DA747_dropoldpairs_just_test.csv')
        # val_image_pair_hrd_black_test747_path8 = os.path.join(image_folder, 'merged_annotation', date,
        #                                             f'aug_train_pair_labels_padgroup_250225_final_rgb_DA747_dropoldpairs_just_test.csv')

        # if args.valdataset == 'cur_jiraissue_hrd_test_on_train_clean':
        #     if img_color == 'rgb':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_rgb_general1, val_image_pair_path_hrd_rgb_general2 ]
        #     elif img_color == 'white':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_white_general1, val_image_pair_path_hrd_white_general2]
        # elif args.valdataset == 'cur_jiraissue_hrd_black_test_DA747':
        #     val_image_pair_path_list = [val_image_pair_hrd_black_test747_path1,val_image_pair_hrd_black_test747_path2,
        #                                 val_image_pair_hrd_black_test747_path3,val_image_pair_hrd_black_test747_path4,
        #                                 val_image_pair_hrd_black_test747_path5,val_image_pair_hrd_black_test747_path6,]
        # elif args.valdataset == 'cur_jiraissue_hrd_black_test_DA747':
        #     val_image_pair_path_list = [val_image_pair_hrd_black_test747_path7,val_image_pair_hrd_black_test747_path8]
                
        # elif args.valdataset == 'cur_jiraissue_hrd_test_on_train_soft':
        #     if img_color == 'rgb':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_rgb_soft1]
        #     elif img_color == 'white':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_white_soft1]
        # elif args.valdataset == 'cur_jiraissue_hrd_test_on_train_softhard':
        #     if img_color == 'rgb':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_rgb_softhard1]
        #     elif img_color == 'white':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_white_softhard1]

        # elif args.valdataset == 'cur_jiraissue_hrd_black_test_clean':
        #     if img_color == 'rgb':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_rgb_supply_clean1, val_image_pair_path_hrd_rgb_supply_clean2]
        #     elif img_color == 'white':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_white_supply_clean1, val_image_pair_path_hrd_white_supply_clean2]
        # elif args.valdataset == 'cur_jiraissue_hrd_black_test_soft':
        #     if img_color == 'rgb':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_rgb_supply_soft1, val_image_pair_path_hrd_rgb_supply_soft2]
        #     elif img_color == 'white':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_white_supply_soft1, val_image_pair_path_hrd_white_supply_soft2]
        # elif args.valdataset == 'cur_jiraissue_hrd_black_test_hard':
        #     if img_color == 'rgb':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_rgb_supply_hard1, val_image_pair_path_hrd_rgb_supply_hard2]
        #     elif img_color == 'white':
        #         val_image_pair_path_list = [val_image_pair_path_hrd_white_supply_hard1, val_image_pair_path_hrd_white_supply_hard2]


