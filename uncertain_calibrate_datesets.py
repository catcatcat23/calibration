import os

import numpy as np
import pandas as pd

from glob import glob
from scipy.stats import entropy

def process_model_predict_results(dir_path, dataset, save_name):
    csvs_path = os.path.join(dir_path, dataset)
    os.makedirs(os.path.join(csvs_path, 'results'), exist_ok=True)
    results_df = pd.DataFrame(columns=['ref_image', 'insp_image', 'material_id', 'cad_hash', 'binary_y'])
    
    for csv_path in glob(os.path.join(csvs_path, '*.csv')):
        cur_df = pd.read_csv(csv_path)
        bi_probs = cur_df[['model_bi_ok_conf', 'model_bi_ng_conf']].values
        cur_df['bi_entropy'] = [entropy(row, base=2) for row in bi_probs]

        multi_class_columns = [col for col in cur_df.columns if 'multi_class' in col]
        multi_class_probs = cur_df[multi_class_columns].astype(float).values
        cur_df['multi_entropy'] = [entropy(row, base=2) for row in multi_class_probs]

        cur_results_df = cur_df[['ref_image', 'insp_image', 'material_id', 'cad_hash', 'binary_y']].copy()
        csv_name = os.path.basename(csv_path).split('.csv')[0]
        cur_results_df[f'{csv_name}_bi_entropy'] = cur_df['bi_entropy'].values
        cur_results_df[f'{csv_name}_multi_entropy'] = cur_df['multi_entropy'].values

        results_df = pd.merge(
            results_df,
            cur_results_df,
            on=['ref_image', 'insp_image', 'material_id', 'cad_hash', 'binary_y'],
            how='outer'  # 可改为 'inner' 若你只想保留完全匹配的行
        )
    
    results_save_csv = os.path.join(csvs_path, 'results', save_name)
    results_df.to_csv(results_save_csv)

def filter_outlier_by_std(datas, dist = 1.5):
    mean = np.mean(datas)
    std = np.std(datas)

    filtered = [x for x in datas if abs(x - mean) <= dist * std]

    return filtered

def _sample_data_by_bi_entropy(sample_ori_df, sample_nums, per_material_sample_lower_limit):
    sample_ori_gp_df = sample_ori_df.groupby(['material_id', 'cad_hash'])
    ori_total_nums = len(sample_ori_df)

    sample_ok_df_list = []
    for gp_keys, infos in sample_ori_gp_df:
        cur_gp_total_nums = len(infos)
        cur_sample_nums = cur_gp_total_nums / ori_total_nums * sample_nums
        
        bi_values = infos['bi_mean_entropy'].values
        bom_values = infos['bom_mean_entropy'].values

        bi_min_values = bi_values.min()
        bi_max_values = bi_values.max()

        cur_gp_bi_entropy_mean = np.mean(bi_values)
        cur_gp_bi_entropy_std = np.std(bi_values)

        dist = 1
        cur_bi_left_upper = cur_gp_bi_entropy_mean
        cur_bi_right_lower = cur_gp_bi_entropy_mean

        while cur_bi_left_upper >= bi_min_values or cur_bi_right_lower <= bi_max_values:
            cur_bi_left_upper = cur_gp_bi_entropy_mean - (dist - 1) * cur_gp_bi_entropy_std
            cur_bi_left_lower = cur_gp_bi_entropy_mean - dist * cur_gp_bi_entropy_std

            cur_bi_right_lower = cur_gp_bi_entropy_mean + (dist - 1) * cur_gp_bi_entropy_std
            cur_bi_right_upper = cur_gp_bi_entropy_mean + dist * cur_gp_bi_entropy_std

            left_mask = (infos['bi_mean_entropy'] > cur_bi_left_lower) & (infos['bi_mean_entropy'] <= cur_bi_left_upper)
            infos.loc[left_mask, 'bi_dist_label'] = -dist

            right_mask = (infos['bi_mean_entropy'] > cur_bi_right_lower) & (infos['bi_mean_entropy'] <= cur_bi_right_upper)
            infos.loc[right_mask, 'bi_dist_label'] = dist
            dist += 1

        infos_sorted = infos.sort_values(by='bi_dist_label').reset_index(drop=True)
        sampled_infos_list = []

        for label, group_df in infos_sorted.groupby('bi_dist_label'):
            group_size = len(group_df)
            if group_size < per_material_sample_lower_limit:
                sampled = group_df
            else:
                ratio = group_size / cur_gp_total_nums  
                sample_num = max(per_material_sample_lower_limit, min(round(cur_sample_nums * ratio), group_size)) 
                sampled = group_df.sample(n=sample_num, random_state=42)
            sampled_infos_list.append(sampled)

        final_sampled_df = pd.concat(sampled_infos_list).reset_index(drop=True)
        sample_ok_df_list.append(final_sampled_df)

    final_sample_df = pd.concat(sample_ok_df_list).reset_index(drop=True)
    return final_sample_df


import numpy as np
import pandas as pd
from numba import jit
from concurrent.futures import ThreadPoolExecutor

@jit(nopython=True)
def calculate_dist_labels(bi_values, mean, std):
    """Numba加速的熵值区间标签计算"""
    labels = np.zeros(len(bi_values), dtype=np.int32)
    dist = 1
    while True:
        # 计算当前距离的左右边界
        left_upper = mean - (dist - 1) * std
        left_lower = mean - dist * std
        right_lower = mean + (dist - 1) * std
        right_upper = mean + dist * std
        
        # 标记区间
        left_mask = (bi_values > left_lower) & (bi_values <= left_upper)
        right_mask = (bi_values > right_lower) & (bi_values <= right_upper)
        
        # 终止条件：当前距离无数据点
        if not (np.any(left_mask) | np.any(right_mask)):
            break
            
        labels[left_mask] = -dist
        labels[right_mask] = dist
        dist += 1
    return labels

def process_group(args):
    """处理单个材料分组的抽样逻辑"""
    gp_keys, infos, global_params = args
    sample_nums, per_material_sample_lower_limit, ori_total_nums = global_params
    
    # 向量化计算
    bi_values = infos['bi_mean_entropy'].to_numpy()
    mean, std = np.mean(bi_values), np.std(bi_values)
    
    # 生成距离标签
    infos = infos.assign(bi_dist_label=calculate_dist_labels(bi_values, mean, std))
    
    # 计算当前组抽样数量
    cur_gp_total_nums = len(infos)
    cur_sample_nums = cur_gp_total_nums / ori_total_nums * sample_nums
    
    # 分层抽样
    samples = []
    for label, group in infos.groupby('bi_dist_label', sort=False):
        n = len(group)
        if n < per_material_sample_lower_limit:
            samples.append(group)
        else:
            ratio = n / cur_gp_total_nums
            n_sample = max(
                per_material_sample_lower_limit,
                min(round(cur_sample_nums * ratio), n)
            )
            samples.append(group.sample(n=n_sample, random_state=42))
    
    return pd.concat(samples) if samples else pd.DataFrame()

def _sample_data_by_bi_entropy_optimized(sample_ori_df, sample_nums, per_material_sample_lower_limit):
    """优化后的主函数"""
    # 预计算全局参数
    ori_total_nums = len(sample_ori_df)
    global_params = (sample_nums, per_material_sample_lower_limit, ori_total_nums)
    
    # 并行处理各组
    with ThreadPoolExecutor() as executor:
        futures = []
        for gp_keys, infos in sample_ori_df.groupby(['material_id', 'cad_hash']):
            futures.append(executor.submit(process_group, (gp_keys, infos, global_params)))
        
        # 收集结果
        results = [f.result() for f in futures]

    return pd.concat(results).reset_index(drop=True)

def sample_data_by_material_and_entropy(df, sample_ok_times = 4, sample_ok_lower_limit_ratio = 0.5, per_material_sample_lower_limit = 10):
    ok_df = df[~df['binary_y']]
    ng_df = df[df['binary_y']]

    total_ok_nums = len(ok_df)
    total_ng_nums = len(ng_df)

    sample_ok_nums = max(total_ng_nums * sample_ok_times, sample_ok_lower_limit_ratio * total_ok_nums)
    sample_ok_df = _sample_data_by_bi_entropy_optimized(ok_df, sample_ok_nums, per_material_sample_lower_limit)
    sample_ng_df = _sample_data_by_bi_entropy_optimized(ng_df, total_ng_nums, per_material_sample_lower_limit)

    sample_df = pd.concat([sample_ok_df, sample_ng_df], ignore_index=True).reset_index(drop=True)
    return sample_df


# 向量化计算函数
def vectorized_filter_mean(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    row_means = np.mean(data, axis=1, keepdims=True)
    row_stds = np.std(data, axis=1, keepdims=True)
    mask = (data >= row_means - k * row_stds) & (data <= row_means + k * row_stds)
    filtered_data = np.where(mask, data, np.nan)
    return np.nanmean(filtered_data, axis=1)

def get_uncertain_test_datasets(ori_uncertain_csv, filter_uncertain_csv, entropy_csv, save_csv):
    ori_df = pd.read_csv(ori_uncertain_csv)
    ori_uncertain_df = ori_df[ori_df['confidence'] == 'uncertain']

    filter_df = pd.read_csv(filter_uncertain_csv)
    filter_valid_df = filter_df[filter_df['confidence'] != 'BAD_PAIR']
    
    entropy_df = pd.read_csv(entropy_csv)

    bi_entropy_columns = [col for col in entropy_df.columns if 'bi_entropy' in col]
    bom_entropy_columns = [col for col in entropy_df.columns if 'multi_entropy' in col]
    entropy_df['bi_mean_entropy'] = vectorized_filter_mean(entropy_df[bi_entropy_columns].values)
    entropy_df['bom_mean_entropy'] = vectorized_filter_mean(entropy_df[bom_entropy_columns].values)

    # 过滤后保留的原始uncertain数据，因为过滤后的csv中这些uncertain变成了certain，无法区分
    valid_uncertain_df = pd.merge(
        ori_uncertain_df,
        filter_valid_df[['ref_image', 'insp_image']],  # 只保留用于匹配的列
        on=['ref_image', 'insp_image'],
        how='inner'
    )

    entropy_matched_uncertain_df = pd.merge(
        entropy_df,
        valid_uncertain_df[['ref_image', 'insp_image']],  # 只保留用于匹配的列
        on=['ref_image', 'insp_image'],
        how='inner'
    )

    entropy_unmatched_uncertain_df = pd.merge(
        entropy_df,
        valid_uncertain_df[['ref_image', 'insp_image']],
        on=['ref_image', 'insp_image'],
        how='left',
        indicator=True
    ).query('_merge == "left_only"').drop(columns=['_merge'])

    if not entropy_unmatched_uncertain_df.empty:
        sample_df = sample_data_by_material_and_entropy(entropy_unmatched_uncertain_df)
        final_df = pd.concat([entropy_matched_uncertain_df, sample_df], ignore_index=True).reset_index(drop=True)
    else:
        final_df = entropy_matched_uncertain_df
    
    # final_df.to_csv(save_csv)

    # 从原始清洗过的数据中挑出对应的pair
    select_test_df_from_final_df = pd.merge(
        filter_valid_df,
        final_df[['ref_image', 'insp_image']], 
        on=['ref_image', 'insp_image'],
        how='inner'
    )

    select_test_df_from_final_df.to_csv(save_csv)
if __name__ == "__main__":
    dir_path =  '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/singlepad'
    dataset =  'all_black'
    save_name = 'all_black_results.csv'

    # process_model_predict_results(dir_path, dataset, save_name)

    ori_uncertain_csv = '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/singlepad/before_pos_uncertain/ori_uncertain_csv/all_singlepad_rgb_uncertain_before_pos_black_test.csv'
    filter_uncertain_csv = '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/singlepad/before_pos_uncertain/ori_uncertain_csv/all_singlepad_rgb_uncertain_before_pos_black_test_update.csv'
    
    entropy_csv =  '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/singlepad/before_pos_uncertain/results/before_pos_uncertain_results.csv'
    save_csv = '/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification/singlepad/before_pos_uncertain/results/before_pos_uncertain_sample_results.csv'
    get_uncertain_test_datasets(ori_uncertain_csv, filter_uncertain_csv, entropy_csv, save_csv)