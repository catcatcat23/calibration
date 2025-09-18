import os
import pandas as pd

def generate_defect_dir_csv(csv_list, defect_dirs,save_path):
    csv_df = pd.concat([pd.read_csv(x) for x in csv_list], ignore_index=True)
    # 过滤出 defect_dir 在 defect_dirs 列表中的行
    filtered_df = csv_df[csv_df['defect_dir'].isin(defect_dirs)]
    filtered_df = filtered_df.drop_duplicates(['ref_image', 'insp_image']).reset_index(drop=True)
    # 保存结果到新CSV文件
    filtered_df.to_csv(save_path, index=False)

def parse_bash_param(bash_param):
    if ',' in bash_param or ' ' in bash_param or \
        (len(bash_param[-1]) == 1 and len(bash_param[0]) == 1):
        bash_params = ''.join(bash_param).split(',')
    else:
        bash_params = bash_param
    
    return bash_params

def load_pair_csv(aug_train_pair_data_filenames, defect_class_considered, defect_code_link, label_confidence, defect_dirs = None):
    Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = [], [], [], []
    # aug_train_pair_data_df_raw = pd.concat([pd.read_csv(f, index_col=0) for f in aug_train_pair_data_filenames]).reset_index(drop=True)
    
    for f in aug_train_pair_data_filenames:
        aug_train_pair_data_df_raw = pd.read_csv(f, index_col=0)
        if label_confidence == 'certain':
            aug_train_pair_data_df_certain = aug_train_pair_data_df_raw[
                (aug_train_pair_data_df_raw['confidence'] == 'certain') | (
                            aug_train_pair_data_df_raw['confidence'] == 'unchecked')].copy()
            print(f'aug pair certain = {len(aug_train_pair_data_df_certain)} out of {len(aug_train_pair_data_df_raw)}')
            aug_train_pair_data_df_raw = aug_train_pair_data_df_certain
        aug_train_pair_data_df = aug_train_pair_data_df_raw[[y in defect_class_considered for
                                                            y in aug_train_pair_data_df_raw['insp_y_raw']]].copy().reset_index(drop=True)
        if defect_dirs is not None:
            print(f'过滤defect dir 前共有数据： {len(aug_train_pair_data_df)}')

            if 'defect_dir' in aug_train_pair_data_df.columns:
                defects_df = []
                for defect_dir in defect_dirs:
                    defects_df.append(aug_train_pair_data_df[aug_train_pair_data_df['defect_dir'] == defect_dir])
                aug_train_pair_data_df = pd.concat(defects_df, ignore_index=True).reset_index(drop=True)
                print(f'过滤defect dir 后共有数据： {len(aug_train_pair_data_df)}')
            else:
                print(f'{f}不需要defect_dir过滤')

            
        n_aug = len(aug_train_pair_data_df)
        print(f' aug with {n_aug} image pairs')
        for i, datum in aug_train_pair_data_df.iterrows():
            Xtrain_resampled.append([datum['ref_image'], datum['insp_image']])
            insp_y_raw = datum['insp_y_raw']
            insp_y = defect_code_link[insp_y_raw]
            ytrain_resampled.append([datum['ref_y'], insp_y])
            ybinary_resampled.append(datum['binary_y'])
            material_train_resampled.append(str(datum['material_id']))
    
    return Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled

def save_traintestval_infos(Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled, annotation_folder_mnt, region):
    train_pair_data = [x + ytrain_resampled[i] + ytrain_resampled[i] + [ybinary_resampled[i], material_train_resampled[i]]
                        for i, x in enumerate(Xtrain_resampled)]
    train_pair_data_df = pd.DataFrame(train_pair_data,
                                columns=['ref_image', 'insp_image', 'ref_y', 'insp_y', 'ref_y_raw', 'insp_y_raw',
                                            'binary_y', 'material_id'])
    train_pair_data_df.to_csv(os.path.join(annotation_folder_mnt, f'annotation_train_pair_labels_{region}.csv'))

def process_csv_files(aug_train_pair_data_filenames, args, label_confidence, defect_class_considered, defect_code_link, datasplit, drop_dup = False):
    aug_train_pair_data_df_raw = get_data_df(aug_train_pair_data_filenames, args.data_type)

    if label_confidence == 'certain':
        aug_train_pair_data_df_certain = aug_train_pair_data_df_raw[
            (aug_train_pair_data_df_raw['confidence'] == 'certain') | (
                        aug_train_pair_data_df_raw['confidence'] == 'unchecked')].copy()
        print(f'aug {datasplit} pair certain = {len(aug_train_pair_data_df_certain)} out of {len(aug_train_pair_data_df_raw)}')
        aug_train_pair_data_df_raw = aug_train_pair_data_df_certain
    aug_pair_data_df = aug_train_pair_data_df_raw[[y in defect_class_considered for
                                                        y in aug_train_pair_data_df_raw['insp_y_raw']]].copy().reset_index(drop=True)
    aug_pair_data_df['insp_y'] = [defect_code_link[yraw] for yraw in aug_pair_data_df['insp_y_raw']]
    
    if drop_dup:
        aug_pair_data_df = aug_pair_data_df.drop_duplicates(['ref_image', 'insp_image']).reset_index(drop=True)

    n_aug = len(aug_pair_data_df)
    print(f'aug {datasplit} with {n_aug} imag_pairs')
    Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = [], [], [], []
    for i, datum in aug_pair_data_df.iterrows():
        Xtrain_resampled.append([datum['ref_image'], datum['insp_image']])
        insp_y_raw = datum['insp_y_raw']
        insp_y = defect_code_link[insp_y_raw]
        ytrain_resampled.append([datum['ref_y'], insp_y])
        ybinary_resampled.append(datum['binary_y'])
        material_train_resampled.append(datum['material_id'])

    return Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled, aug_pair_data_df
    # train_pair_data = [x + ytrain_resampled[i] + ytrain_resampled[i] + [ybinary_resampled[i], material_train_resampled[i]]
    #                     for i, x in enumerate(Xtrain_resampled)]

def get_data_df(aug_train_pair_data_filenames, data_type):
    aug_train_2d_pair_data_list = []
    aug_train_3d_pair_data_list = []

    for train_csv_path in aug_train_pair_data_filenames:
        cur_train_df = pd.read_csv(train_csv_path, index_col=0)
        print(f"loading from {train_csv_path} || nums={len(cur_train_df)}")

        if 'feature_set_name' not in cur_train_df.columns:
            aug_train_2d_pair_data_list.append(cur_train_df)
        
        else:
            cur_3d_train_df = cur_train_df[cur_train_df['feature_set_name'] != 'default']
            cur_2d_train_df = cur_train_df[cur_train_df['feature_set_name'] == 'default']

            aug_train_3d_pair_data_list.append(cur_3d_train_df)
            aug_train_2d_pair_data_list.append(cur_2d_train_df)


    if data_type == '2d':
        aug_train_pair_data_df_raw = pd.concat(aug_train_2d_pair_data_list).reset_index(drop=True)
    elif data_type == '3d':
        aug_train_pair_data_df_raw = pd.concat(aug_train_3d_pair_data_list).reset_index(drop=True)
    else:
        aug_train_pair_data_df_raw = pd.concat(aug_train_3d_pair_data_list + aug_train_2d_pair_data_list).reset_index(drop=True)
    
    return aug_train_pair_data_df_raw

def filter_from_df(aug_train_pair_data, data_type):
    aug_train_2d_pair_data_list = []
    aug_train_3d_pair_data_list = []

    # for train_csv_path in aug_train_pair_data_filenames:
    #     cur_train_df = pd.read_csv(train_csv_path, index_col=0)
    if 'feature_set_name' not in aug_train_pair_data.columns:
        aug_train_2d_pair_data_list.append(aug_train_pair_data)
    
    else:
        cur_3d_train_df = aug_train_pair_data[(aug_train_pair_data['feature_set_name'] == 'default3D')| 
                                              (aug_train_pair_data['feature_set_name'] == 'temp3D')]
        
        cur_2d_train_df = aug_train_pair_data[(aug_train_pair_data['feature_set_name'] != 'default3D') & 
                                              (aug_train_pair_data['feature_set_name'] != 'temp3D')]
        
        aug_train_3d_pair_data_list.append(cur_3d_train_df)
        aug_train_2d_pair_data_list.append(cur_2d_train_df)


    if data_type == '2d':
        aug_train_pair_data_df_raw = pd.concat(aug_train_2d_pair_data_list).reset_index(drop=True)
    elif data_type == '3d':
        aug_train_pair_data_df_raw = pd.concat(aug_train_3d_pair_data_list).reset_index(drop=True)
    else:
        aug_train_pair_data_df_raw = pd.concat(aug_train_3d_pair_data_list + aug_train_2d_pair_data_list).reset_index(drop=True)
    
    return aug_train_pair_data_df_raw