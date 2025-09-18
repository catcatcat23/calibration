import os
import argparse

import pandas as pd

from algo_test_utils import get_test_all_csvs, get_test_df
from copy import deepcopy

def main(args):
    date = args.date
    part_name = args.part_name
    img_color = args.img_color

    if part_name == 'body' or part_name == 'componnet':
        image_folder = '/mnt/dataset/xianjianming/data_clean_white/'
    elif part_name == 'debug':
        image_folder = '/mnt/dataset/xianjianming/debug/'
    else:
        image_folder = '/mnt/dataset/xianjianming/data_clean/'
    val_image_pair_path_list = get_test_all_csvs(image_folder, date,  part_name, args.valdataset, img_color)


    tmp_list= []
    [tmp_list.append(x) for x in val_image_pair_path_list  if x not in tmp_list]
    val_image_pair_path_list = deepcopy(tmp_list)
    print(args.valdataset, val_image_pair_path_list)

    val_image_pair_data_list = []
    # val_image_pair_data_list = [pd.read_csv(path, index_col=0) for path in val_image_pair_path_list]
    for path in val_image_pair_path_list:
        csv_name = os.path.basename(path).split('.')[0]
        csv_df = pd.read_csv(path)
        csv_df['csv_name'] = csv_name
        if 'confidence' not in csv_df.columns:
            csv_df['confidence'] = 'certain'

        val_image_pair_data_list.append(csv_df)

    val_image_pair_data_raw = pd.concat(val_image_pair_data_list)  # 
    val_image_pair_data_raw = val_image_pair_data_raw.drop_duplicates(subset=['ref_image', 'insp_image']).reset_index(drop=True)
 
    special_df = pd.read_csv(os.path.join(image_folder, 'merged_annotation', date, args.special_csv))
    special_dir_df = special_df[special_df['defect_dir'] == args.special_dir]

    result_df = val_image_pair_data_raw.merge(special_dir_df[['ref_image', 'insp_image']], on=['ref_image', 'insp_image'], how='inner')
    result_df['defect_dir'] = args.special_dir
    result_df.to_csv(os.path.join(image_folder, 'merged_annotation', date,args.special_df_save_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate special data')   #padgroup1.17select  padgroup1.16TGselect padgroup1.14impl
    parser.add_argument('--part_name', default=f'padgroup', type=str)   #padgroup
    parser.add_argument('--date', default=f'241022', type=str) 
    parser.add_argument('--img_color', default=f'white', type=str, choices=['rgb', 'white']) 
    parser.add_argument('--valdataset', default=f'padgroup_test_white_all_data', type=str)

    parser.add_argument('--special_csv', default=f'padgroup_test_white_all_data_padgroup1.19.3tp1finalselect_all_update.csv', type=str) 
    parser.add_argument('--special_dir', default=f'NG_EXTREME', type=str) 
    parser.add_argument('--special_df_save_name', default=f'padgroup_extrem_white_ng.csv', type=str) 

    args = parser.parse_args()
    main(args)
    
