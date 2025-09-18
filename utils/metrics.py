import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from scipy.special import lambertw, softmax
import matplotlib.pyplot as plt
import matplotlib

import os

def get_ECE_calculator(args):
    # 计算校准误差
    if args.ECE_calcu_type == 'ECE':
        return ECEer().calculate_ece_mce
    elif args.ECE_calcu_type == 'Ada_ECE':
        return ECEer().calculate_ada_ece_mce
    elif args.ECE_calcu_type == 'Classwise_ECE':
        return ECEer().calculate_classwise_ece_mce        
    elif args.ECE_calcu_type == 'Classwise_Ada_ECE':
        return ECEer().calculate_classwise_ada_ece_mce 
    elif  args.ECE_calcu_type == 'Multi_ECE':
        return  ECEer().calculate_multiclass_ece_mce
    elif  args.ECE_calcu_type == 'KDE':
        return KDEer().calculate_overall_ece_kde
    elif  args.ECE_calcu_type == 'Classwise_KDE':
        return KDEer().calculate_classwise_ece_kde
    elif  args.ECE_calcu_type == 'Multi_KDE':
        return KDEer().calculate_multiclass_ece_mce_kde
    elif  args.ECE_calcu_type == 'TrueKDE':
        return KDEer().calculate_trueclass_ece_kde  

    
def plot_multi_acc_conf(bin_list, acc_mean_list, confidences_mean_list, save_path=None):
    bin_list = np.array(bin_list)
    bin_centers = (bin_list[:-1] + bin_list[1:]) / 2
    bin_widths = np.diff(bin_list)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左边 y 轴：acc_mean_list 柱状图
    bars1 = ax1.bar(bin_centers - 0.015, acc_mean_list, width=bin_widths * 0.4, color='skyblue', label='Accuracy per bin', align='center')
    ax1.set_ylabel('Accuracy', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xlabel('Confidence bin')
    ax1.set_ylim(0, 1.05)

    # 标注 Accuracy 数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.3f}", ha='center', va='bottom', fontsize=8, color='blue')

    # 右边 y 轴：confidences_mean_list 柱状图
    ax2 = ax1.twinx()
    bars2 = ax2.bar(bin_centers + 0.015, confidences_mean_list, width=bin_widths * 0.4, color='salmon', label='Confidence per bin', align='center')
    ax2.set_ylabel('Confidence', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    ax2.set_ylim(0, 1.05)

    # 标注 Confidence 数值
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.3f}", ha='center', va='bottom', fontsize=8, color='darkred')

    plt.title('Accuracy vs Confidence per Bin')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()



def plot_acc_conf(acc_mean_list, confidences_mean_list, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(confidences_mean_list, acc_mean_list, marker='o', linestyle='-', color='b', label="Accuracy vs. Confidence")

    for x, y in zip(confidences_mean_list, acc_mean_list):
        plt.text(x, y+0.005, f"({x:.3f}, {y:.3f})", ha='center', va='bottom', fontsize=10, color='black')

    # 添加轴标签和标题
    plt.xlabel("Confidence")  # (置信度)
    plt.ylabel("Accuracy")  #  (准确度)
    plt.title("Accuracy vs. Confidence Curve")
    plt.legend()
    plt.grid(True)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300)
        # print(f"图像已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_bin_ece(bin_list, ece_list, save_path=None):
    bin_centers = (np.array(bin_list[:-1]) + np.array(bin_list[1:])) / 2  # 计算中心点

    plt.figure(figsize=(8, 6))

    # 绘制柱状图
    plt.bar(bin_centers, ece_list, width=np.diff(bin_list), edgecolor='black', alpha=0.6, label="ECE per bin")

    # 在柱状图中心点绘制折线
    plt.plot(bin_centers, ece_list, marker='o', linestyle='-', color='r', label="ECE Trend")

   # 在每个点上方显示数值
    for x, y in zip(bin_centers, ece_list):
        plt.text(x, y+0.005, f"({x:.3f}, {y:.3f})", ha='center', va='bottom', fontsize=10, color='black')

    # 添加轴标签和标题
    plt.xlabel("Bins")
    plt.ylabel("ECE Value")
    plt.title("ECE per Bin with Trend Line")
    plt.legend()
    plt.grid(True)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300)
        # print(f"图像已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_bin_nums(bin_list, bin_num_list, save_path=None):
    bin_centers = (np.array(bin_list[:-1]) + np.array(bin_list[1:])) / 2  # 计算中心点

    plt.figure(figsize=(8, 6))

    # 绘制柱状图
    plt.bar(bin_centers, bin_num_list, width=np.diff(bin_list), edgecolor='black', alpha=0.6, label="sample nums per bin")

    # 在柱状图中心点绘制折线
    plt.plot(bin_centers, bin_num_list, marker='o', linestyle='-', color='r', label="sample nums Trend")

   # 在每个点上方显示数值
    for x, y in zip(bin_centers, bin_num_list):
        plt.text(x, y, f"({x:.3f}, {y})", ha='center', va='bottom', fontsize=10, color='black')

    # 添加轴标签和标题
    plt.xlabel("Bins")
    plt.ylabel("sample nums")
    plt.title("sample per Bin with Trend Line")
    plt.legend()
    plt.grid(True)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300)
        # print(f"图像已保存至: {save_path}")
    else:
        plt.show() 
    plt.close()

# Calibration error scores in the form of loss metrics
class ECEer():

    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=10):
        super(ECEer, self).__init__()
        self.nbins = n_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def _calculate_ece_mce(self, logits, confidences, accuracies, bin_lowers, bin_uppers):
        ece = torch.zeros(1, device=logits.device)
        mce = torch.zeros(1, device=logits.device)

        bin_list = []
        bin_nums = []
        bin_confs = []
        acc_mean_list = []
        confidences_mean_list = []
        ece_list = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()  # confidence是该执行都范围内所有预测类别的均值
                gap = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += gap * prop_in_bin
                mce = max(mce, gap)

                if len(bin_list) == 0:
                    bin_list.append(bin_lower.item())
                bin_list.append(bin_upper.item())
                bin_nums.append(in_bin.sum().item())
                acc_mean_list.append(accuracy_in_bin.item())
                confidences_mean_list.append(avg_confidence_in_bin.item())
                ece_list.append(gap.item())

                confidences_in_bin = confidences[in_bin] # 在这个范围内所有的置信度，但是这些置信度也可能是其他类别的置信度
                accuracies_in_bin = accuracies[in_bin] # 在这个范围内所有的预测类别，True为预测正确，False为预测错误
                bin_confs.append(confidences[in_bin])

        return ece, mce, bin_list, bin_nums, acc_mean_list, confidences_mean_list, ece_list
    
    def _calculate_multiclass_ece_mce(self, softmaxes, labels, bin_lowers, bin_uppers, defect_decode_dict):
        ece = torch.zeros(1, device=softmaxes.device)
        mce = torch.zeros(1, device=softmaxes.device)
        max_confidences, max_predictions = torch.max(softmaxes, 1)
        unique_labels = torch.unique(labels)

        bin_list = []
        bin_nums = []
        bin_confs = []
        acc_mean_dict = {}
        confidences_mean_dict = {}
        ece_list = []
        for unq_label in unique_labels:
            acc_mean_dict[defect_decode_dict[unq_label.item()]] = []
            confidences_mean_dict[defect_decode_dict[unq_label.item()]] = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = max_confidences.gt(bin_lower.item()) * max_confidences.le(bin_upper.item())
            cur_bin_weight = in_bin.float().mean()
            
            if cur_bin_weight.item() > 0:
                prop_in_bin = softmaxes[in_bin]
                labels_in_bin = labels[in_bin]

                gaps = 0
                cur_acc_mean = 0
                for unq_label in unique_labels:
                    cur_label_acc = (labels_in_bin == unq_label).float().mean()
                    cur_label_probs_avg = prop_in_bin[:, int(unq_label.item())].mean()
                    gap = abs(cur_label_acc - cur_label_probs_avg)
                    gaps += gap
                    cur_acc_mean += cur_label_acc

                    acc_mean_dict[defect_decode_dict[unq_label.item()]].append(cur_label_acc.item())
                    confidences_mean_dict[defect_decode_dict[unq_label.item()]].append(cur_label_probs_avg.item())

                ece += gaps * cur_bin_weight
                mce = max(mce, gaps)

                if len(bin_list) == 0:
                    bin_list.append(bin_lower.item())
                bin_list.append(bin_upper.item())
                bin_nums.append(in_bin.sum().item())
                # acc_mean_list.append(cur_acc_mean)
                # confidences_mean_list.append(avg_confidence_in_bin.item())
                ece_list.append(gaps.item())

        return ece, mce, bin_list, bin_nums, acc_mean_dict, confidences_mean_dict, ece_list

    def _get_multi_acc_conf(self, labels, confidences, predictions):
        unique_labels = torch.unique(labels)

        label_lists = []
        confidences_list = []
        accuracies_list = []

        label_flat = labels.view(-1)
        assert label_flat.shape == confidences.shape
        for cur_label in unique_labels:
            cur_label_idx = label_flat.eq(cur_label)
            cur_confidences = confidences[cur_label_idx]
            cur_predictions = predictions[cur_label_idx]
            cur_accuracies = cur_predictions.eq(cur_label)

            confidences_list.append(cur_confidences)
            accuracies_list.append(cur_accuracies)
            label_lists.append(cur_label)
        return confidences_list, accuracies_list, label_lists

    def _histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))  # interp(x, xp, fp）利用已知数据点（xp, fp）对输入 x 处的函数值进行线性插值  

    # 正常f1score中R的权重是β**2/(1 + β**2), P的权重是1/(1 + β**2)
    # 本身误报就要更重要，因为ok数据量更大，所以平衡状态下 bgama就应该小于1 
    # ogama强调漏报，要大于bgama,
    # fgama强调误报，要小于bgama, 需要注意的点在于β的作用是成平方放大的，此时已经小于了，所以在调节fgama时候，要更小心一点
    # self_gama是计算ok，ng各自的f1-score，所以应该更关注recall

    @classmethod
    def calculate_threshold(cls, logits_bos, labels_bos, logits_bom, labels_bom, defect_decode, 
                            thresh_range, default_thres = 0.8, thres_dt = 0.02, bgama = 0.5, ogama = None, fgama = None, self_gama = 1.5):
        assert logits_bos.shape[-1] == 2, f"仅用二分类进行阈值筛选"
        output_bos_np_softmax = softmax(logits_bos, axis=1)
        output_bom_np_softmax = softmax(logits_bom, axis=1)

        labels_bos_np = np.ascontiguousarray(labels_bos.reshape(-1, 1))
        labels_bom_np = np.ascontiguousarray(labels_bom.reshape(-1, 1))

        num_points = int((thresh_range[1] - thresh_range[0]) / thres_dt) + 1
        thresh_array = np.linspace(thresh_range[0], thresh_range[1], num=num_points)

        # 判断 default_thres 是否在 thresh_array 中
        if not np.isin(default_thres, thresh_array):
            thresh_array = np.append(thresh_array, default_thres)  # 添加 0.8
            thresh_array = np.sort(thresh_array)  # 排序，保持数组有序

        ok_total_nums = (labels_bos == 0).sum()
        ng_total_nums = (labels_bos == 1).sum()
        total_nums = ok_total_nums + ng_total_nums

        if ogama is None:
            if ng_total_nums == 0:
                ogama = bgama *(1 + 0.1)  # 假设真实的ng数据占总数据的10%
            else:
                ogama = bgama *(1.5 - ng_total_nums / total_nums)

        if fgama is None:
            if ok_total_nums == 0:
                fgama = bgama *(0.9)
            else:
                fgama = bgama * abs(0.5 - ok_total_nums / total_nums)

        wf_score, wo_score, bfo_score = 0, 0, 0
        wf_thres, wo_thres, bfo_thres = 0, 0, 0

        ok_self_f1_score, ng_self_f1_score = 0, 0
        ok_self_thres, ng_self_thres = 0, 0

        best_bos_acc, best_bom_acc = 0, 0
        best_bos_acc_thres = 0.5
        best_bom_acc_thres = 0.5

        data_infos = {'ok_recall': {}, 'ng_recall': {}, 
                    'ok_precision': {}, 'ng_precision': {}, 
                    'f1_score_ok':{}, 'f1_score_ng':{},
                    'acc_binary':{}, 'acc_mclass':{},
                    }
        for candidate_thres in thresh_array:
            indices = np.where(output_bos_np_softmax[:, 1] < candidate_thres)
            output_bos_np_softmax_new = output_bos_np_softmax.copy()
            output_bom_np_softmax_new = output_bom_np_softmax.copy()

            output_bos_np_softmax_new[indices, 1] = 0
            output_bom_np_softmax_new[indices, 0] = 2

            # thres_confidences, thres_predictions = torch.max(output_bos_np_softmax_new, 1)

            acc_binary_new, acc_mclass_new, binary_multimetrics_np_new, mclass_multimetrics_np_new, store_df_th = \
            evaluate_val_resultsnew(output_bos_np_softmax_new, output_bom_np_softmax_new,
                                    labels_bos_np, labels_bom_np,
                                    defect_decode, return_pr=False, show_res = False
                                    )
            
            if acc_binary_new >= best_bos_acc:
                best_bos_acc = acc_binary_new
                best_bos_acc_thres = candidate_thres

            if acc_mclass_new >= best_bom_acc:
                best_bom_acc = acc_mclass_new
                best_bom_acc_thres = candidate_thres

            # cur_thres_ok_recall = -1 if np.isnan(binary_multimetrics_np_new[0]['recall']) else binary_multimetrics_np_new[0]['recall']
            # cur_thres_ng_recall = -1 if np.isnan(binary_multimetrics_np_new[1]['recall']) else binary_multimetrics_np_new[1]['recall']
            # cur_thres_ok_f1_score = -1 if np.isnan(binary_multimetrics_np_new[0]['f1_score']) else binary_multimetrics_np_new[0]['f1_score']
            # cur_thres_ng_f1_score = -1 if np.isnan(binary_multimetrics_np_new[1]['f1_score']) else binary_multimetrics_np_new[1]['f1_score']
            # cur_thres_ok_precision = -1 if np.isnan(binary_multimetrics_np_new[0]['precision']) else binary_multimetrics_np_new[0]['precision']
            # cur_thres_ng_precision = -1 if np.isnan(binary_multimetrics_np_new[1]['precision']) else binary_multimetrics_np_new[1]['precision']

            # cur_ok_self_f1 = (1+self_gama**2)*(cur_thres_ok_precision * cur_thres_ok_recall) / (self_gama**2 * cur_thres_ok_precision + cur_thres_ok_recall)
            # cur_ng_self_f1 = (1+self_gama**2)*(cur_thres_ng_precision * cur_thres_ng_recall) / (self_gama**2 * cur_thres_ng_precision + cur_thres_ng_recall)

            # # 误报要小 == ng的precision要高, 或者ok的recall要高（主要还是ng的precision，因为precision代表了该类型的所有预测，但recall只是部分预测） -> 
            # # ng的f1_score计算中precision占比要大，ok的f1_score计算中recall占比要大->ng的beta要小于1, ng的beta要大于1，
            # # 但是不能只考虑ng的precision和ok的recall，因为即使这两个都高，可能ok的precison和ng的recall很低，即漏报严重

            # # 漏报要小 == 和上面相反，要ok的precision高, 且ng的recall要高
            # # fp_weight_f1反映误报，fp = false prediction； op_weight_f1反映漏报，op = omit prediction
            # fp_weight_f1 = (ok_total_nums / total_nums) * cur_thres_ok_recall + (ng_total_nums / total_nums) * cur_thres_ng_precision
            # op_weight_f1 = (ok_total_nums / total_nums) * cur_thres_ok_precision + (ng_total_nums / total_nums) * cur_thres_ng_recall

            # # gama大于1, 更关注op_weight_f1， 小于1，更关注fp_weight_f1
            # cur_bfo_score = (1+bgama**2)*(fp_weight_f1 * op_weight_f1) / (bgama**2 * fp_weight_f1 + op_weight_f1)
            # cur_wf_score = (1+fgama**2)*(fp_weight_f1 * op_weight_f1) / (fgama**2 * fp_weight_f1 + op_weight_f1)
            # cur_wo_score = (1+ogama**2)*(fp_weight_f1 * op_weight_f1) / (ogama**2 * fp_weight_f1 + op_weight_f1)

            # if cur_ok_self_f1 >= ok_self_f1_score:
            #     ok_self_f1_score = cur_ok_self_f1
            #     ok_self_thres = candidate_thres

            # if cur_ng_self_f1 >= ng_self_f1_score:
            #     ng_self_f1_score = cur_ng_self_f1
            #     ng_self_thres = candidate_thres

            # if cur_wf_score >= wf_score:
            #     wf_score = cur_wf_score
            #     wf_thres = candidate_thres

            # if cur_wo_score >= wo_score:
            #     wo_score = cur_wo_score
            #     wo_thres = candidate_thres            

            # if cur_bfo_score >= bfo_score:
            #     bfo_score = cur_bfo_score
            #     bfo_thres = candidate_thres

            data_infos['acc_binary'][candidate_thres] = acc_binary_new
            data_infos['acc_mclass'][candidate_thres] = acc_mclass_new
            # data_infos['ok_recall'][candidate_thres] = cur_thres_ok_recall
            # data_infos['ng_recall'][candidate_thres] = cur_thres_ng_recall
            # data_infos['f1_score_ok'][candidate_thres] = cur_thres_ok_f1_score
            # data_infos['f1_score_ng'][candidate_thres] = cur_thres_ng_f1_score
            # data_infos['ok_precision'][candidate_thres] = cur_thres_ok_precision
            # data_infos['ng_precision'][candidate_thres] = cur_thres_ng_precision

        # data_infos['ok_self_thres'] = ok_self_thres
        # data_infos['ng_self_thres'] = ng_self_thres
        # data_infos['ok_self_f1_score'] = ok_self_f1_score
        # data_infos['ng_self_f1_score'] = ng_self_f1_score

        data_infos['best_bos_acc'] = best_bos_acc
        data_infos['best_bom_acc'] = best_bom_acc
        data_infos['best_bos_acc_thres'] = best_bos_acc_thres
        data_infos['best_bom_acc_thres'] = best_bom_acc_thres 

        # data_infos['wf_score'] = wf_score
        # data_infos['wo_score'] = wo_score
        # data_infos['bfo_score'] = bfo_score
        # data_infos['wf_thres'] = wf_thres
        # data_infos['wo_thres'] = wo_thres
        # data_infos['bfo_thres'] = bfo_thres
        

        return data_infos
    

    def _calculate_classwise_ada_ece_mce(self,
        cur_confidences, cur_accuracies, cur_label, logits, defect_decode_dict, save_path,
        vis_acc_conf=True, vis_bin_ece=True, if_save=True, ece_calculator=None
    ):
        n, cur_bin_boundaries = np.histogram(
            cur_confidences.detach(), 
            ece_calculator._histedges_equalN(cur_confidences.detach())
        )
        bin_lowers = cur_bin_boundaries[:-1]
        bin_uppers = cur_bin_boundaries[1:]

        cur_ece, cur_mce, bin_list, bin_nums, acc_mean_list, confidences_mean_list, ece_list = \
            ece_calculator._calculate_ece_mce(
                logits, cur_confidences, cur_accuracies, bin_lowers, bin_uppers
            )

        label_name = defect_decode_dict[cur_label.item()]
        print(f"{label_name} bin_list = {bin_list}")

        # 保存图像
        bin_nums_img_path = os.path.join(save_path, f'classwise_ada_ece_mce_total_bin_nums_{label_name}.png')
        plot_bin_nums(bin_list, bin_nums, bin_nums_img_path)

        if vis_acc_conf and if_save:
            acc_conf_img_path = os.path.join(save_path, f'classwise_ada_ece_mce_total_acc_conf_{label_name}.png')
            plot_acc_conf(acc_mean_list, confidences_mean_list, acc_conf_img_path)

        if vis_bin_ece and if_save:
            bin_ece_img_path = os.path.join(save_path, f'classwise_ada_ece_mce_total_bin_ece_{label_name}.png')
            plot_bin_ece(bin_list, ece_list, bin_ece_img_path)

        return label_name, cur_ece, cur_mce
    
    
    def calculate_classwise_ada_ece_mce(self, logits, labels, save_path = None, vis_acc_conf = False, vis_bin_ece = False, defect_decode_dict = None):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        # accuracies = predictions.eq(labels)
        multi_ece = {}
        multi_mce = {}

        confidences_list, accuracies_list, label_lists = self._get_multi_acc_conf(labels, confidences, predictions)
        if_save = save_path is not None

        for cur_confidences, cur_accuracies, cur_label in zip(confidences_list, accuracies_list, label_lists):
            n, cur_bin_boundaries = np.histogram(cur_confidences.detach(), self._histedges_equalN(cur_confidences.detach()))
            self.bin_lowers = cur_bin_boundaries[:-1]
            self.bin_uppers = cur_bin_boundaries[1:]

            cur_ece, cur_mce, bin_list, bin_nums, acc_mean_list, confidences_mean_list, ece_list = self._calculate_ece_mce(logits, cur_confidences, cur_accuracies, self.bin_lowers, self.bin_uppers)
            multi_ece[defect_decode_dict[cur_label.item()]] = cur_ece
            multi_mce[defect_decode_dict[cur_label.item()]] = cur_mce
            print(f"{defect_decode_dict[cur_label.item()]} bin_list = {bin_list}")

            bin_nums_img_path = os.path.join(save_path, f'classwise_ada_ece_mce_total_bin_nums_{defect_decode_dict[cur_label.item()]}.png')
            plot_bin_nums(bin_list, bin_nums, bin_nums_img_path)

            if vis_acc_conf and if_save:
                acc_conf_img_path = os.path.join(save_path, f'classwise_ada_ece_mce_total_acc_conf_{defect_decode_dict[cur_label.item()]}.png')
                plot_acc_conf(acc_mean_list, confidences_mean_list, acc_conf_img_path)

            if vis_bin_ece and if_save:
                bin_ece_img_path = os.path.join(save_path, f'classwise_ada_ece_mce_total_bin_ece_{defect_decode_dict[cur_label.item()]}.png')
                plot_bin_ece(bin_list, ece_list, bin_ece_img_path)

        return multi_ece, multi_mce

    def calculate_ada_ece_mce(self, logits, labels, save_path = None, vis_acc_conf = False, vis_bin_ece = False, defect_decode_dict = None):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)

        labels_flat = labels.view(-1)
        assert predictions.shape == labels_flat.shape
        accuracies = predictions.eq(labels_flat)

        n, bin_boundaries = np.histogram(confidences.detach(), self._histedges_equalN(confidences.detach()))

        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece_dict = {}
        mce_dict = {}

        ece, mce, bin_list, bin_nums, acc_mean_list, confidences_mean_list, ece_list = self._calculate_ece_mce(logits, confidences, accuracies, self.bin_lowers, self.bin_uppers)
        ece_dict['total_ada_ece'] = ece
        mce_dict['total_ada_mce'] = mce
        print(f"bin_list = {bin_list}")

        bin_nums_img_path = os.path.join(save_path, f'ada_ece_mce_total_bin_nums.png')
        plot_bin_nums(bin_list, bin_nums, bin_nums_img_path)

        if_save = save_path is not None
        if vis_acc_conf and if_save:
            acc_conf_img_path = os.path.join(save_path, 'ada_ece_mce_total_acc_conf.png')
            plot_acc_conf(acc_mean_list, confidences_mean_list, acc_conf_img_path)

        if vis_bin_ece and if_save:
            bin_ece_img_path = os.path.join(save_path, 'ada_ece_mce_total_bin_ece.png')
            plot_bin_ece(bin_list, ece_list, bin_ece_img_path)

        return ece_dict, mce_dict


    def calculate_multiclass_ece_mce(self, logits, labels, save_path = None, vis_acc_conf = False, 
                                     vis_bin_ece = False, defect_decode_dict = None):
        softmaxes = F.softmax(logits, dim=1)

        ece_dict = {}
        mce_dict = {}

        # multiclass_ece下acc_mean_list, confidences_mean_list没有意义
        ece, mce, bin_list, bin_nums, acc_mean_dict, confidences_mean_dict, ece_list = \
        self._calculate_multiclass_ece_mce(softmaxes, labels, self.bin_lowers, self.bin_uppers, defect_decode_dict)

        ece_dict['multiclass_ece'] = ece
        mce_dict['multiclass_mce'] = mce

        # bin_nums_img_path = os.path.join(save_path, f'multiclass_ece_mce_total_bin_nums.png')
        # plot_bin_nums(bin_list, bin_nums, bin_nums_img_path)

        if_save = save_path is not None
        if vis_acc_conf and if_save:
            for defect_type, defect_acc_list in acc_mean_dict.items():
                acc_conf_img_path = os.path.join(save_path, f'multiclass_ece_mce_acc_conf_{defect_type}.png')
                defect_conf_list = confidences_mean_dict[defect_type]
                plot_multi_acc_conf(bin_list, defect_acc_list, defect_conf_list, acc_conf_img_path)


        # if vis_bin_ece and if_save:
        #     bin_ece_img_path = os.path.join(save_path, 'multiclass_ece_mce_total_bin_ece.png')
            # plot_bin_ece(bin_list, ece_list, bin_ece_img_path)

        return ece_dict, mce_dict

    def calculate_classwise_ece_mce(self, logits, labels, save_path = None,vis_acc_conf = False, vis_bin_ece = False, defect_decode_dict = None):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        multi_ece = {}
        multi_mce = {}

        confidences_list, accuracies_list, label_lists = self._get_multi_acc_conf(labels, confidences, predictions)
        if_save = save_path is not None
       
        for cur_confidences, cur_accuracies, cur_label in zip(confidences_list, accuracies_list, label_lists):
            cur_ece, cur_mce, bin_list, bin_nums, acc_mean_list, confidences_mean_list, ece_list = self._calculate_ece_mce(logits, cur_confidences, cur_accuracies, self.bin_lowers, self.bin_uppers)
            multi_ece[defect_decode_dict[cur_label.item()]] = cur_ece
            multi_mce[defect_decode_dict[cur_label.item()]] = cur_mce
            # print(f"{defect_decode_dict[cur_label.item()]} bin_list = {bin_list}")

            bin_nums_img_path = os.path.join(save_path, f'classwise_ece_mce_total_bin_nums_{defect_decode_dict[cur_label.item()]}.png')
            plot_bin_nums(bin_list, bin_nums, bin_nums_img_path)

            if vis_acc_conf and if_save:
                acc_conf_img_path = os.path.join(save_path, f'classwise_ece_mce_total_acc_conf_{defect_decode_dict[cur_label.item()]}.png')
                plot_acc_conf(acc_mean_list, confidences_mean_list, acc_conf_img_path)

            if vis_bin_ece and if_save:
                bin_ece_img_path = os.path.join(save_path, f'classwise_ece_mce_total_bin_ece_{defect_decode_dict[cur_label.item()]}.png')
                plot_bin_ece(bin_list, ece_list, bin_ece_img_path)

        return multi_ece, multi_mce
    
    def calculate_ece_mce(self, logits, labels, save_path = None, vis_acc_conf = False, vis_bin_ece = False, defect_decode_dict = None):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        labels_flat = labels.view(-1)
        assert predictions.shape == labels_flat.shape
        accuracies = predictions.eq(labels_flat)
        ece_dict = {}
        mce_dict = {}

        ece, mce, bin_list, bin_nums, acc_mean_list, confidences_mean_list, ece_list = self._calculate_ece_mce(logits, confidences, accuracies, self.bin_lowers, self.bin_uppers)
        ece_dict['total_ece'] = ece
        mce_dict['total_mce'] = mce

        bin_nums_img_path = os.path.join(save_path, f'ece_mce_total_bin_nums.png')
        plot_bin_nums(bin_list, bin_nums, bin_nums_img_path)

        if_save = save_path is not None
        if vis_acc_conf and if_save:
            acc_conf_img_path = os.path.join(save_path, 'ece_mce_total_acc_conf.png')
            plot_acc_conf(acc_mean_list, confidences_mean_list, acc_conf_img_path)

        if vis_bin_ece and if_save:
            bin_ece_img_path = os.path.join(save_path, 'ece_mce_total_bin_ece.png')
            plot_bin_ece(bin_list, ece_list, bin_ece_img_path)

        return ece_dict, mce_dict
    

# -*- coding: utf-8 -*-
import os
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F


# -*- coding: utf-8 -*-
import os
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F


class KDEer:
    """
    KDE-based ECE with precise LOO and boundary reflection (stabilized).

    提供四种口径：
      - overall（top-1）
      - classwise（按被预测类）
      - trueclass（按真实类 one-vs-rest）
      - multiclass（样本上对 K 维误差求和）

    关键改动（更稳）：
      1) 带宽 h = max(scale * Silverman, h_min)   —— 缓解尖峰/锯齿
      2) 反射权重加系数 λ，且仅在离边界 τ·h 内启用 —— 边缘不再“立柱”
      3) LOO 时的自权重与上述反射保持一致（强度与启用区间相同）

    兼容性：
      - 仍接受 logits（内部 softmax）
      - 返回字典的键保持不变：'multiclass_ece' / 'multiclass_mce' / 'overall_ece' / 'overall_mce'
    """

    def __init__(self,
                 kde_grid_points: int = 201,
                 kde_bandwidth: Optional[float] = None,
                 kde_reflect: bool = False,
                 eps: float = 1e-8,
                 # 新增稳化超参
                 kde_bandwidth_scale: float = 2.0,   # Silverman 带宽放大倍数
                 kde_bandwidth_min: float = 0.06,    # 带宽下界
                 kde_reflect_strength: float = 0.5,  # 反射强度 λ ∈ [0,1]
                 kde_reflect_window: float = 2.5):   # 反射只在距边界 < τ·h 时启用
        self.kde_grid_points = int(kde_grid_points)
        self.kde_bandwidth = kde_bandwidth
        self.kde_reflect = bool(kde_reflect)
        self.eps = float(eps)

        self.kde_bandwidth_scale = float(kde_bandwidth_scale)
        self.kde_bandwidth_min = float(kde_bandwidth_min)
        self.kde_reflect_strength = float(kde_reflect_strength)
        self.kde_reflect_window = float(kde_reflect_window)

    # ---------------- 基础工具：带宽 / 核函数 / 网格 / 插值 ----------------
    def _calc_bandwidth(self, probs: torch.Tensor) -> float:
        """
        带宽 h（更稳的版本）：
          h0 = 1.06 * std * n^(-1/5)
          h  = max(scale * h0, h_min)
        - 若显式传入 self.kde_bandwidth，则直接使用该值。
        """
        if self.kde_bandwidth is not None:
            return float(self.kde_bandwidth)

        pk_all = probs.reshape(-1).detach()
        std = float(pk_all.std(unbiased=True))
        n = pk_all.numel()
        # Silverman 规则
        h0 = 1.06 * (std if std > 0.0 else 0.5) * (n ** (-1.0 / 5.0))
        # 放大 + 下界
        h = max(h0 * self.kde_bandwidth_scale, self.kde_bandwidth_min)
        return float(h)

    @staticmethod
    def _gauss_kernel(x: torch.Tensor, h: float) -> torch.Tensor:
        """高斯核（归一化常数在比值里会约掉）。"""
        return torch.exp(-0.5 * (x / h) ** 2)

    def _grid(self, device: torch.device) -> torch.Tensor:
        """[0,1] 上均匀网格（G 点）"""
        return torch.linspace(0.0, 1.0, steps=self.kde_grid_points, device=device)

    def _interp_two_sided(self, grid_kg: torch.Tensor, t_nk: torch.Tensor) -> torch.Tensor:
        """
        在线性网格上做线性插值（矢量化）：
            grid_kg: [K,G]，每类在 G 个网格点的值
            t_nk   : [N,K]，每样本/类的“网格坐标”（实数，0~G-1）
        返回 [N,K]
        """
        G = grid_kg.shape[1]
        t = t_nk.clamp(0, G - 1 - 1e-8)
        i0 = t.floor().long()            # [N,K]
        i1 = (i0 + 1).clamp(max=G - 1)   # [N,K]
        alpha = (t - i0.float()).clamp(0.0, 1.0)  # [N,K]

        # 通过转置让 gather 的 batch 维度匹配
        g0 = grid_kg.gather(1, i0.T).T   # [N,K]
        g1 = grid_kg.gather(1, i1.T).T   # [N,K]
        return (1 - alpha) * g0 + alpha * g1

    # ------- 统一构造带“温和反射”的核权重矩阵 W（所有估计都用它） -------
    def _kernel_weights_with_reflect(self,
                                     c_grid: torch.Tensor,   # [G]
                                     samples: torch.Tensor,  # [M]
                                     h: float) -> torch.Tensor:
        """
        返回 W: [G, M]
        反射仅在 |c-边界| < τ·h 内启用，并乘系数 λ。
        """
        diff = c_grid.view(-1, 1) - samples.view(1, -1)   # [G,M]
        W = self._gauss_kernel(diff, h)

        if self.kde_reflect:
            tau = self.kde_reflect_window
            lam = self.kde_reflect_strength

            # 仅在靠近边界的网格点启用反射项
            left_on = (c_grid.view(-1, 1) < tau * h).float()
            right_on = (c_grid.view(-1, 1) > 1.0 - tau * h).float()

            W_left = self._gauss_kernel(c_grid.view(-1, 1) - (-samples).view(1, -1), h)
            W_right = self._gauss_kernel(c_grid.view(-1, 1) - (2.0 - samples).view(1, -1), h)

            W = W + lam * left_on * W_left + lam * right_on * W_right

        return W

    # --------------------------- 绘图（legend 版） ---------------------------
    @staticmethod
    def _plot_reliability_curve(
        c_grid,
        pi_curves,
        save_path=None,
        title='',
        hist_vals=None,
        mode='multiclass',         # 'multiclass' | 'classwise'
        class_names=None,          # dict/list（键是 id 时映射）
        debug_show=False,
    ):
        try:
            import matplotlib
            if not debug_show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            if pi_curves is None or (isinstance(pi_curves, dict) and len(pi_curves) == 0):
                print(f"Skip plotting {title}: no pi_curves data")
                return
            if (not debug_show) and (save_path is None):
                print("Skip plotting: save_path is None and debug_show=False")
                return
            if (save_path is not None) and (not debug_show):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=140)
            c_grid_np = np.asarray(c_grid, dtype=float)

            # ideal 对角线
            ax.plot([0, 1], [0, 1], '--', lw=1, color='gray', label='ideal')

            if mode == 'multiclass':
                y = np.asarray(pi_curves, dtype=float)
                ax.plot(c_grid_np, y, lw=2, label='KDE reliability')
                if hist_vals is not None and len(hist_vals) > 1:
                    ax2 = ax.twinx()
                    ax2.hist(np.asarray(hist_vals), bins=20, range=(0, 1), alpha=0.25)
                    ax2.set_yticks([])

            elif mode == 'classwise':
                # 将键排序，保证颜色/顺序稳定
                if all(isinstance(k, (int, np.integer)) for k in pi_curves.keys()):
                    items = sorted(pi_curves.items(), key=lambda kv: int(kv[0]))
                else:
                    items = sorted(pi_curves.items(), key=lambda kv: str(kv[0]))

                N = len(items)
                cmap = plt.cm.get_cmap('tab10' if N <= 10 else 'tab20', max(N, 10))

                for idx, (k, curve) in enumerate(items):
                    curve_np = np.asarray(curve, dtype=float)
                    color = cmap(idx)
                    # 类名解析
                    if isinstance(k, (str, np.str_)):
                        name = str(k)
                    else:
                        if class_names is not None:
                            if isinstance(class_names, dict):
                                name = class_names.get(int(k), f'class_{k}')
                            elif isinstance(class_names, (list, tuple)) and int(k) < len(class_names):
                                name = class_names[int(k)]
                            else:
                                name = f'class_{k}'
                        else:
                            name = f'class_{k}'
                    ax.plot(c_grid_np, curve_np, lw=2, color=color, label=name)

                # 可选：各类直方图
                if isinstance(hist_vals, dict) and len(hist_vals) > 0:
                    ax2 = ax.twinx()
                    for idx, (k, _) in enumerate(items):
                        vals = hist_vals.get(k, None)
                        if vals is not None and len(vals) > 1:
                            ax2.hist(np.asarray(vals), bins=20, range=(0, 1), alpha=0.12,
                                     color=cmap(idx))
                    ax2.set_yticks([])

            # 统一的坐标/外观
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('confidence c')
            ax.set_ylabel('P(correct | c)')
            ax.set_title(title)
            ax.legend(loc='lower right', fontsize=8)

            fig.tight_layout()
            if debug_show:
                plt.show()
            else:
                fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"Plot reliability failed: {e}")

    # ======================== 1) Multi-class ========================
    @torch.no_grad()
    def calculate_multiclass_ece_mce_kde(self,
                                         logits: torch.Tensor,   # [N,K]
                                         labels: torch.Tensor,   # [N]
                                         save_path: Optional[str] = None,
                                         vis_acc_conf: bool = False,
                                         vis_bin_ece: bool = False,
                                         defect_decode_dict: Optional[Dict[int, str]] = None
                                         ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        device = logits.device
        probs = F.softmax(logits, dim=1).clamp(self.eps, 1 - self.eps)  # [N,K]
        labels = labels.long()
        N, K = probs.shape

        # 每类带宽（更稳）
        h_list = [float(self._calc_bandwidth(probs[:, k])) for k in range(K)]
        h_k = torch.tensor(h_list, device=device, dtype=probs.dtype)     # [K]

        # 网格与每类 KDE
        G = self.kde_grid_points
        c_grid = self._grid(device)                                      # [G]
        PI = torch.empty(K, G, device=device, dtype=probs.dtype)
        NUM = torch.empty_like(PI)
        DEN = torch.empty_like(PI)

        for k in range(K):
            p_k = probs[:, k]                          # [N]
            y_k = (labels == k).float()                # [N]
            hk = float(h_k[k])

            W = self._kernel_weights_with_reflect(c_grid, p_k, hk)       # [G,N]
            num_g = (W * y_k.view(1, -1)).sum(dim=1)                     # [G]
            den_g = W.sum(dim=1) + self.eps                               # [G]
            PI[k] = (num_g / den_g).clamp(self.eps, 1 - self.eps)
            NUM[k], DEN[k] = num_g, den_g

        # 插值到每个样本自己的 p_{ik} 上，做精确 LOO
        t = probs * (G - 1)                                              # [N,K]
        num_at_p = self._interp_two_sided(NUM, t)                        # [N,K]
        den_at_p = self._interp_two_sided(DEN, t)                        # [N,K]

        # 自身权重（与反射一致的强度与启用区间）
        w_self = torch.ones_like(probs)                                   # [N,K]
        if self.kde_reflect:
            lam = self.kde_reflect_strength
            tau = self.kde_reflect_window
            hk = h_k.view(1, -1)                                          # [1,K] for broadcast
            left_on = (probs < tau * hk).float()
            right_on = (probs > 1.0 - tau * hk).float()
            w_self = (w_self
                      + lam * left_on * torch.exp(-0.5 * (2.0 * probs / hk) ** 2)
                      + lam * right_on * torch.exp(-0.5 * (2.0 * (1.0 - probs) / hk) ** 2))

        y_onehot = F.one_hot(labels, num_classes=K).float()
        num_loo = (num_at_p - y_onehot * w_self).clamp_min(self.eps)
        den_loo = (den_at_p - w_self).clamp_min(self.eps)
        pi_loo = (num_loo / den_loo).clamp(self.eps, 1 - self.eps)

        abs_gap = (pi_loo - probs).abs()
        gaps_sum = abs_gap.sum(dim=1)
        ece_multi = gaps_sum.mean()
        mce_multi = gaps_sum.max()

        ece_dict = {'multiclass_ece': ece_multi}
        mce_dict = {'multiclass_mce': mce_multi}

        # 可选可视化：micro vs macro
        if vis_acc_conf and (save_path is not None):
            import numpy as np
            os.makedirs(save_path, exist_ok=True)

            num_sum = NUM.sum(dim=0)
            den_sum = DEN.sum(dim=0).clamp_min(self.eps)
            pi_micro = (num_sum / den_sum).clamp(self.eps, 1 - self.eps)

            pk = probs.detach().reshape(-1).cpu().numpy()
            pk_hist = pk if pk.size <= 200_000 else pk[np.random.choice(pk.size, 200_000, replace=False)]

            self._plot_reliability_curve(
                c_grid.detach().cpu().tolist(),
                pi_micro.detach().cpu().tolist(),
                os.path.join(save_path, "kde_multi_micro.png"),
                title="KDE reliability (multi, micro)",
                hist_vals=pk_hist.tolist(),
                mode='multiclass'
            )

            pi_macro = PI.mean(dim=0).clamp(self.eps, 1 - self.eps)
            curves = {0: pi_micro.detach().cpu().tolist(),
                      1: pi_macro.detach().cpu().tolist()}
            names = {0: 'micro (pooled i,k)', 1: 'macro (avg over classes)'}
            self._plot_reliability_curve(
                c_grid.detach().cpu().tolist(),
                curves,
                os.path.join(save_path, "kde_multi_micro_vs_macro.png"),
                title="KDE reliability (multi: micro vs macro)",
                hist_vals=None,
                mode='classwise',
                class_names=names
            )

        return ece_dict, mce_dict

    # ======================== 2) Class-wise（pred） ========================
    @torch.no_grad()
    def calculate_classwise_ece_kde(self,
                                    logits: torch.Tensor,    # [N,K]
                                    labels: torch.Tensor,    # [N]
                                    save_path: Optional[str] = None,
                                    vis_acc_conf: bool = False,
                                    vis_bin_ece: bool = False,
                                    defect_decode_dict: Optional[Dict[int, str]] = None
                                    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        device = logits.device
        probs = F.softmax(logits, dim=1).clamp(self.eps, 1 - self.eps)
        labels = labels.long()
        N, K = probs.shape
        preds = probs.argmax(dim=1)

        c_grid = self._grid(device)
        G = self.kde_grid_points

        out_dict: Dict[str, torch.Tensor] = {}
        mce_dummy: Dict[str, torch.Tensor] = {}

        classwise_curves: Dict[str, list] = {}
        classwise_hists: Dict[str, list] = {}

        for k in range(K):
            mask = (preds == k)
            # 类名解析
            if isinstance(defect_decode_dict, dict):
                name = defect_decode_dict.get(k, f'class_{k}')
            elif isinstance(defect_decode_dict, (list, tuple)) and k < len(defect_decode_dict):
                name = defect_decode_dict[k]
            else:
                name = f'class_{k}'

            M = int(mask.sum())
            if M == 0:
                out_dict[name] = torch.tensor(0.0, device=device)
                continue

            p_k = probs[mask, k]                          # [M]
            y_k = (labels[mask] == k).float()             # [M]

            hk = float(self._calc_bandwidth(p_k))
            # 小样本额外放大一点带宽（更稳）
            if M < 50:
                hk *= (50.0 / max(M, 1)) ** (1.0 / 5.0)

            W = self._kernel_weights_with_reflect(c_grid, p_k, hk)   # [G,M]
            num_g = (W * y_k.view(1, -1)).sum(dim=1)                 # [G]
            den_g = W.sum(dim=1) + self.eps                          # [G]
            pi_g = (num_g / den_g).clamp(self.eps, 1 - self.eps)     # [G]

            # 插值 + LOO
            t = (p_k * (G - 1)).clamp(0, G - 1 - 1e-8)
            i0 = t.floor().long()
            i1 = (i0 + 1).clamp(max=G - 1)
            alpha = (t - i0.float()).clamp(0.0, 1.0)

            g0n = num_g.gather(0, i0); g1n = num_g.gather(0, i1)
            g0d = den_g.gather(0, i0); g1d = den_g.gather(0, i1)

            num_at_p = (1 - alpha) * g0n + alpha * g1n
            den_at_p = (1 - alpha) * g0d + alpha * g1d

            # 自权重（与反射一致）
            w_self = torch.ones_like(p_k)
            if self.kde_reflect:
                lam = self.kde_reflect_strength
                tau = self.kde_reflect_window
                left_on = (p_k < tau * hk).float()
                right_on = (p_k > 1.0 - tau * hk).float()
                w_self = (w_self
                          + lam * left_on * torch.exp(-0.5 * (2.0 * p_k / hk) ** 2)
                          + lam * right_on * torch.exp(-0.5 * (2.0 * (1.0 - p_k) / hk) ** 2))

            if M >= 2:
                num_loo = (num_at_p - y_k * w_self).clamp_min(self.eps)
                den_loo = (den_at_p - w_self).clamp_min(self.eps)
            else:
                num_loo = num_at_p.clamp_min(self.eps)
                den_loo = den_at_p.clamp_min(self.eps)

            pi_loo = (num_loo / den_loo).clamp(self.eps, 1 - self.eps)
            ece_k = (pi_loo - p_k).abs().mean()
            out_dict[name] = ece_k

            classwise_curves[name] = pi_g.detach().cpu().tolist()
            classwise_hists[name] = p_k.detach().cpu().tolist()

        if vis_acc_conf and (save_path is not None):
            self._plot_reliability_curve(
                c_grid.detach().cpu().tolist(),
                classwise_curves,
                os.path.join(save_path, "kde_classwise_acc_conf_all.png"),
                title="KDE reliability (classwise all)",
                hist_vals=classwise_hists,
                mode='classwise',
                class_names=defect_decode_dict
            )

        return out_dict, mce_dummy

    # ======================== 3) True-class（one-vs-rest） =================
    @torch.no_grad()
    def calculate_trueclass_ece_kde(self,
                                    logits: torch.Tensor,   # [N,K]
                                    labels: torch.Tensor,   # [N]
                                    save_path: Optional[str] = None,
                                    vis_acc_conf: bool = False,
                                    defect_decode_dict: Optional[Dict[int, str]] = None
                                    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        device = logits.device
        probs = F.softmax(logits, dim=1).clamp(self.eps, 1 - self.eps)
        labels = labels.long()
        N, K = probs.shape

        # 每个真实类的带宽
        hk_list = [float(self._calc_bandwidth(probs[:, k])) for k in range(K)]
        hk = torch.tensor(hk_list, device=device, dtype=probs.dtype).view(1, K)  # [1,K]

        c_grid = self._grid(device)
        G = self.kde_grid_points

        PI = torch.empty((K, G), device=device, dtype=probs.dtype)
        NUM = torch.empty_like(PI)
        DEN = torch.empty_like(PI)

        y_onehot = F.one_hot(labels, num_classes=K).float()

        for k in range(K):
            p_k = probs[:, k]                  # [N]
            y_k = y_onehot[:, k]               # [N]
            hk_k = float(hk[0, k].item())

            W = self._kernel_weights_with_reflect(c_grid, p_k, hk_k)     # [G,N]
            num_g = (W * y_k.view(1, -1)).sum(dim=1)
            den_g = W.sum(dim=1) + self.eps
            PI[k] = (num_g / den_g).clamp(self.eps, 1 - self.eps)
            NUM[k], DEN[k] = num_g, den_g

        # 插值到样本自己的 p_{ik}
        t = probs * (G - 1)
        num_at_p = self._interp_two_sided(NUM, t)
        den_at_p = self._interp_two_sided(DEN, t)

        # 精确 LOO（与反射一致）
        w_self = torch.ones_like(probs)
        if self.kde_reflect:
            lam = self.kde_reflect_strength
            tau = self.kde_reflect_window
            left_on = (probs < tau * hk).float()
            right_on = (probs > 1.0 - tau * hk).float()
            w_self = (w_self
                      + lam * left_on * torch.exp(-0.5 * (2.0 * probs / hk) ** 2)
                      + lam * right_on * torch.exp(-0.5 * (2.0 * (1.0 - probs) / hk) ** 2))

        num_loo = (num_at_p - y_onehot * w_self).clamp_min(self.eps)
        den_loo = (den_at_p - w_self).clamp_min(self.eps)
        pi_loo = (num_loo / den_loo).clamp(self.eps, 1 - self.eps)

        abs_gap = (pi_loo - probs).abs()
        ece_per_class = abs_gap.mean(dim=0)

        out_dict: Dict[str, torch.Tensor] = {}
        for k in range(K):
            name = defect_decode_dict.get(k, f'class_{k}') if defect_decode_dict else f'class_{k}'
            out_dict[name] = ece_per_class[k]

        if vis_acc_conf and (save_path is not None):
            classwise_curves = {}
            classwise_hists = {}
            for k in range(K):
                name = defect_decode_dict.get(k, f'class_{k}') if defect_decode_dict else f'class_{k}'
                classwise_curves[name] = PI[k].detach().cpu().tolist()
                classwise_hists[name] = probs[:, k].detach().cpu().tolist()

            self._plot_reliability_curve(
                c_grid.detach().cpu().tolist(),
                classwise_curves,
                os.path.join(save_path, "kde_trueclass_acc_conf_all.png"),
                title="KDE reliability (true-class all)",
                hist_vals=classwise_hists,
                mode='classwise',
                class_names=defect_decode_dict
            )

        return out_dict, {}

    # ======================== 4) Overall（top-1） ==========================
    @torch.no_grad()
    def calculate_overall_ece_kde(self,
                                  logits: torch.Tensor,     # [N,K]
                                  labels: torch.Tensor,     # [N]
                                  save_path: Optional[str] = None,
                                  vis_acc_conf: bool = False
                                  ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        device = logits.device
        probs = F.softmax(logits, dim=1).clamp(self.eps, 1 - self.eps)
        preds = probs.argmax(dim=1)
        labels = labels.long()

        c = probs.max(dim=1).values                          # [N]
        a = (preds == labels).float()                        # [N]

        hc = float(self._calc_bandwidth(c))

        c_grid = self._grid(device)
        W = self._kernel_weights_with_reflect(c_grid, c, hc)             # [G,N]

        num_g = (W * a.view(1, -1)).sum(dim=1)
        den_g = W.sum(dim=1) + self.eps
        pi_g = (num_g / den_g).clamp(self.eps, 1 - self.eps)

        # 插值 + LOO
        G = self.kde_grid_points
        t = (c * (G - 1)).clamp(0, G - 1 - 1e-8)
        i0 = t.floor().long()
        i1 = (i0 + 1).clamp(max=G - 1)
        alpha = (t - i0.float()).clamp(0.0, 1.0)

        g0n = num_g.gather(0, i0); g1n = num_g.gather(0, i1)
        g0d = den_g.gather(0, i0); g1d = den_g.gather(0, i1)

        num_at_c = (1 - alpha) * g0n + alpha * g1n
        den_at_c = (1 - alpha) * g0d + alpha * g1d

        # 自权重（与反射一致）
        w_self = torch.ones_like(c)
        if self.kde_reflect:
            lam = self.kde_reflect_strength
            tau = self.kde_reflect_window
            left_on = (c < tau * hc).float()
            right_on = (c > 1.0 - tau * hc).float()
            w_self = (w_self
                      + lam * left_on * torch.exp(-0.5 * (2.0 * c / hc) ** 2)
                      + lam * right_on * torch.exp(-0.5 * (2.0 * (1.0 - c) / hc) ** 2))

        num_loo = (num_at_c - a * w_self).clamp_min(self.eps)
        den_loo = (den_at_c - w_self).clamp_min(self.eps)
        pi_loo = (num_loo / den_loo).clamp(self.eps, 1 - self.eps)

        abs_gap = (pi_loo - c).abs()
        ece_overall = abs_gap.mean()
        mce_overall = abs_gap.max()

        ece_dict = {'overall_ece': ece_overall}
        mce_dict = {'overall_mce': mce_overall}

        if vis_acc_conf and (save_path is not None):
            self._plot_reliability_curve(
                c_grid.detach().cpu().tolist(),
                pi_g.detach().cpu().tolist(),
                os.path.join(save_path, "kde_overall_acc_conf.png"),
                title="KDE reliability (overall)",
                hist_vals=c.detach().cpu().tolist(),
                mode='multiclass'
            )

        return ece_dict, mce_dict



# Calibration error scores in the form of loss metrics
class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()  # in_bin是bool值，用于提取bin中的值，prop_in_bin表示该bin中元素占总元素的比例，所以prop_in_bin为0表示该bin中没有元素
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))  # interp(x, xp, fp）利用已知数据点（xp, fp）对输入 x 处的函数值进行线性插值
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

def get_gamma(p=0.2):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
    gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
    gamma = np.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma

ps = [0.2, 0.5]
gammas = [5.0, 3.0]
i = 0
gamma_dic = {}
for p in ps:
    gamma_dic[p] = gammas[i]
    i += 1

class FocalLossAdaptive(nn.Module):
    def __init__(self, gamma=0, size_average=False, device=None):
        super(FocalLossAdaptive, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.device = device

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        gamma = self.get_gamma_list(pt)
        loss = -1 * (1-pt)**gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def compute_ece_mce_multiclass(probs, labels, M=10):
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    for a **multiclass setting**.
    
    probs: (N, K) array of softmax-calibrated probabilities.
    labels: (N,) array of true labels.
    M: Number of bins.
    
    Returns:
        ece: Expected Calibration Error (float)
        mce: Maximum Calibration Error (float)
    """
    N, K = probs.shape  # Number of samples, number of classes
    confidences = np.max(probs, axis=1)  # Take highest class probability
    predictions = np.argmax(probs, axis=1)  # Get predicted class
    correct = (predictions == labels).astype(int)  # 1 if correct, 0 otherwise

    # Fixed-width binning from 0 to 1
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    for i in range(M):
        in_bin = (confidences >= bin_lowers[i]) & (confidences < bin_uppers[i])
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(correct[in_bin])  # Accuracy in bin
            bin_conf = np.mean(confidences[in_bin])  # Average confidence in bin
            gap = abs(bin_acc - bin_conf)  # Difference between accuracy and confidence
            ece += (bin_count / N) * gap  # Weighted contribution to ECE
            mce = max(mce, gap)  # Maximum Calibration Error

    return ece, mce

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


class ContrastiveCosinLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveCosinLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_similarity = F.cosine_similarity(output1, output2, dim=1, eps=1e-8)
        cosine_distance = (1 - cosine_similarity).unsqueeze(1)
        loss_contrastive = torch.mean((1-label) * torch.pow(cosine_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))


        return loss_contrastive


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, weighted=False, label_smooth=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.weighted = weighted
        self.size_average = size_average
        self.label_smooth = label_smooth

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """
        # print("classes", classes)
        one_hot = torch.zeros(labels.size(0), classes)

        # labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, n_classes, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.

        Args:
            target: target in form with [label1, label2, label_batchsize]
            n_classes: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        # print("length", length)
        # print("smooth_fact", smooth_factor)
        one_hot = self._one_hot(target, n_classes, value=1 - smooth_factor)
        one_hot += smooth_factor / n_classes

        return one_hot.to(target.device)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)

        if self.label_smooth > 0:
            smoothed_target = self._smooth_label(target, input.size(1), self.label_smooth)
            logpt = torch.sum(logpt * smoothed_target, dim=1)
        else:
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.weighted:
            # weights for each class
            target_unique = target.unique(sorted=True)
            n_unique = target_unique.size()[0]
            batch_size = target.size()[0]
            target_relative_freq = torch.stack(
                [(n_unique*(target == t).sum() / batch_size) for t in target])
            loss = torch.div(loss, target_relative_freq)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def accuracy(output, target, topk=(1,),  input_type='torch',  class_input=False):
    """Computes the top k accuracy"""
    if input_type == 'torch':
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    elif input_type == 'np':
        k = 1
        if class_input:
            max_k_preds = output
        else:
            max_k_preds = output.argsort(axis=1)[:, -k:][:, ::-1]
        
        # print(f"max_k_preds shape = {max_k_preds.shape}")
        # print(f"target shape = {target.shape}")

        match_array = np.logical_or.reduce(max_k_preds == target, axis=1)
        topk_accuracy = match_array.sum() / match_array.shape[0]
        return topk_accuracy


def compute_scores(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score
    p_score = precision_score(y_true, y_pred)
    r_score = recall_score(y_true, y_pred)
    return p_score, r_score

def predict_afterthresh(y_scores, threholds):
    return (y_scores >= threholds) * 1

def p_r_curve(y_true, y_scores):
    # thresholds = sorted(np.unique(y_scores))
    thresholds = np.linspace(0, 1, 101)
    precisions, recalls, f1_scores = [], [], []
    for thre in thresholds:
        y_pred = predict_afterthresh(y_scores, thre)
        p, r = compute_scores(y_true, y_pred)
        precisions.append(p)
        recalls.append(r)
        if p == 0 and r == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        f1_scores.append(f1)
    # 去掉召回率中末尾重复的情况
    # last_ind = np.searchsorted(recalls[::-1], recalls[0]) + 1
    # precisions = precisions[-last_ind:]
    # recalls = recalls[-last_ind:]
    # thresholds = thresholds[-last_ind:]
    # precisions.append(1)
    # recalls.append(0)
    return precisions, recalls, f1_scores, thresholds

def classk_metric(output_bos, true_label, classk, beta=0.5, input_type='np', class_input=False):
    if input_type == 'np':
        if class_input:
            pred = output_bos.flatten()
        else:
            pred = np.argmax(output_bos, axis=1)

        # compute predicted, true of classk
        if classk == -1:
            # classk = -1 indicate computing metrics for all defects in combined
            # classk = 0 indicate ok sample
            pred_classk = ~(pred==0)
            true_classk = ~(true_label==0)

        else:
            pred_classk = (pred == classk) # 预测为
            true_classk = (true_label == classk)

        # compute true positive, false positive, false negative, true negative of classk
        tp_classk = pred_classk * true_classk
        fp_classk = pred_classk * (~true_classk)
        fn_classk = (~pred_classk) * true_classk
        tn_classk = (~pred_classk) * (~true_classk)

        # compute number of predicted, true, true positive, false positive,
        # false negative, true negative of classk
        npred_classk = pred_classk.sum(0)
        ntrue_classk = true_classk.sum(0)
        ntp_classk = tp_classk.sum(0)
        nfp_classk = fp_classk.sum(0)
        nfalse_classk = (~true_classk).sum(0)

    else:
        _, pred = output_bos.topk(1, 1, True, True)
        pred = pred.t()

        # compute predicted, true of classk
        if classk == -1:
            # classk = -1 indicate computing metrics for all defects in combined
            # classk = 0 indicate ok sample
            pred_classk = ~pred.eq(0)
            true_classk = ~true_label.eq(0)

        else:
            pred_classk = pred.eq(classk)
            true_classk = true_label.eq(classk)

        # compute true positive, false positive, false negative, true negative of classk
        tp_classk = pred_classk * true_classk
        fp_classk = pred_classk * (~true_classk)
        fn_classk = (~pred_classk) * true_classk
        tn_classk = (~pred_classk) * (~true_classk)

        # compute number of predicted, true, true positive, false positive,
        # false negative, true negative of classk
        npred_classk = pred_classk.view(-1).float().sum(0)
        ntrue_classk = true_classk.view(-1).float().sum(0)
        ntp_classk = tp_classk.view(-1).float().sum(0)
        nfp_classk = fp_classk.view(-1).float().sum(0)
        nfalse_classk = (~true_classk).view(-1).float().sum(0)


    # compute precision and recall
    precision = ntp_classk / npred_classk
    recall = ntp_classk / (ntrue_classk)
    f1_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    omission_rate = nfp_classk/nfalse_classk
    # print(f'precision@class={classk}: {precision}')
    # print(f'recall@class={classk}: {recall}')
    return precision, recall, f1_score, omission_rate, ntrue_classk

def multiclass_multimetric(output_bos, true_label, classk_list, input_type, class_input=False):
    n_total = 0
    weight_f1_score = 0
    res = {}
    for classk in classk_list+[-1]:
        if classk == 0:  # ok需要更关注precision, 降低漏报
            beta = 0.5
        else:
            beta = 1.5  # ng更关注recall, 降低漏报

        precision, recall, f1_score, omission_rate, ntrue_classk = \
            classk_metric(output_bos, true_label, classk, beta, input_type, class_input)


        weight_f1_score += f1_score*ntrue_classk
        n_total += ntrue_classk
        classk_res = {'precision': precision,
                      'recall': recall,
                      'f1_score': f1_score,
                      'omission_rate': omission_rate
                      }
        res[classk] = classk_res

    weight_f1_score = weight_f1_score/n_total
    res['weight_f1_score'] = weight_f1_score
    return res

def multiclass_multimetric_custom(output_bos, true_label, classk_list, classk_beta, input_type, class_input=False):
    n_total = 0
    weight_f1_score = 0
    res = {}
    for classk in classk_list+[-1]:
        # if classk == 0:  # beta < 0更关注precision, beta > 0更关注recall
        #     beta = 0.5
        # else:
        #     beta = 1.5
        beta = classk_beta[classk]
        precision, recall, f1_score, omission_rate, ntrue_classk = \
            classk_metric(output_bos, true_label, classk, beta, input_type, class_input)


        weight_f1_score += f1_score*ntrue_classk
        n_total += ntrue_classk
        classk_res = {'precision': precision,
                      'recall': recall,
                      'f1_score': f1_score,
                      'omission_rate': omission_rate
                      }
        res[classk] = classk_res

    weight_f1_score = weight_f1_score/n_total
    res['weight_f1_score'] = weight_f1_score
    return res

def evaluate_val_results(output_bos_np_softmax, output_bom_np_softmax, binary_labels, insp_labels, defect_decode,
                         class_input=False, return_pr=False):

    # compute accuracy
    acc_binary = accuracy(output_bos_np_softmax, binary_labels, input_type='np', class_input=class_input)
    acc_mclass = accuracy(output_bom_np_softmax, insp_labels, input_type='np', class_input=class_input)

    print(f'binary accuracy = {acc_binary:.5f}')
    print(f'mclass accuracy = {acc_mclass:.5f}')

    # compute other metrics
    # check other metrics:
    # binary_label =0: ok, =1: defect
    binary_class_list = ['ok', 'ng']
    binary_class_number = list(range(len(binary_class_list)))

    binary_multimetrics_np = multiclass_multimetric(output_bos_np_softmax, binary_labels.flatten(), binary_class_number, 'np', class_input=class_input)
    n_binary_dic = {bi_c: np.sum(binary_labels.flatten() == i) for i, bi_c in enumerate(binary_class_list)}

    print('== Binary output ==')
    print(f'weighted f1 score = ' + str(binary_multimetrics_np['weight_f1_score']))
    for i, bi_c in enumerate(binary_class_list):
        multimetric_value = binary_multimetrics_np[i]
        multimetric_value_str_list = [f'{key}:{value:.3f}' for key, value in multimetric_value.items()]
        print(f'{bi_c} (n={n_binary_dic[bi_c]}): ' + ', '.join(multimetric_value_str_list))


    mclass_class_number = list(range(len(defect_decode.keys())))
    mclass_multimetrics_np = multiclass_multimetric(output_bom_np_softmax, insp_labels.flatten(), mclass_class_number, 'np', class_input=class_input)

    print('== Multiclass output ==')
    print(f'weighted f1 score = ' + str(mclass_multimetrics_np['weight_f1_score']))

    n_mclass_dic = {}
    for i, mi_c in defect_decode.items():
        multimetric_value = mclass_multimetrics_np[i]
        multimetric_value_str_list = [f'{key}:{value:.3f}' for key, value in multimetric_value.items()]
        n_mclass_dic[mi_c] = np.sum(insp_labels.flatten() == i)
        print(f'{mi_c} (n={n_mclass_dic[mi_c]}): ' + ', '.join(multimetric_value_str_list))

    # compute pr curve
    if not class_input and return_pr:
        y_scores = output_bos_np_softmax[:, 1]
        y_true = binary_labels.flatten()
        precision_list, recall_list, f1score_list, thresholds_list = p_r_curve(y_true, y_scores)

    if return_pr:
        return acc_binary, acc_mclass, binary_multimetrics_np, mclass_multimetrics_np, precision_list, recall_list, f1score_list, thresholds_list
    else:
        return acc_binary, acc_mclass, binary_multimetrics_np, mclass_multimetrics_np

def evaluate_val_resultsnew(output_bos_np_softmax, output_bom_np_softmax, binary_labels, insp_labels, defect_decode,
                         class_input=False, return_pr=False, show_res = True):
    res_dict = {}
    # compute accuracy
    acc_binary = accuracy(output_bos_np_softmax, binary_labels, input_type='np', class_input=class_input)
    acc_mclass = accuracy(output_bom_np_softmax, insp_labels, input_type='np', class_input=class_input)
    res_string_to_print = f'binary accuracy = {acc_binary:.5f}\n'
    res_string_to_print += f'mclass accuracy = {acc_mclass:.5f}\n'
    res_dict['binary_accuracy'] = acc_binary
    res_dict['mclass_accuracy'] = acc_binary

    # compute other metrics
    # check other metrics:
    # binary_label =0: ok, =1: defect
    binary_class_list = ['ok', 'ng']
    binary_class_number = list(range(len(binary_class_list)))
    res_string_to_print += '== Binary output ==\n'
    binary_multimetrics_np = multiclass_multimetric(output_bos_np_softmax, binary_labels.flatten(), binary_class_number,
                                                    'np', class_input=class_input)
    n_binary_dic = {bi_c: np.sum(binary_labels.flatten() == i) for i, bi_c in enumerate(binary_class_list)}

    res_string_to_print += f'weighted f1 score = ' + str(binary_multimetrics_np['weight_f1_score']) + '\n'
    res_dict['binary_weighted_f1_score'] = binary_multimetrics_np['weight_f1_score']
    for i, bi_c in enumerate(binary_class_list):
        multimetric_value = binary_multimetrics_np[i]
        multimetric_value_str_list = [f'{key}:{value:.3f}' for key, value in multimetric_value.items()]
        res_string_to_print += f'{bi_c} (n={n_binary_dic[bi_c]}): ' + ', '.join(multimetric_value_str_list) + '\n'
        for key, value in multimetric_value.items():
            res_dict[f'bi_{bi_c}_{key}'] = value

    mclass_class_number = list(range(len(defect_decode.keys())))
    mclass_multimetrics_np = multiclass_multimetric(output_bom_np_softmax, insp_labels.flatten(), mclass_class_number,
                                                    'np', class_input=class_input)

    res_string_to_print += '== Multiclass output == \n'
    res_string_to_print += f'weighted f1 score = ' + str(mclass_multimetrics_np['weight_f1_score']) + '\n'
    res_dict['mclass_weighted_f1_score'] = mclass_multimetrics_np['weight_f1_score']

    n_mclass_dic = {}
    for i, mi_c in defect_decode.items():
        multimetric_value = mclass_multimetrics_np[i]
        multimetric_value_str_list = [f'{key}:{value:.3f}' for key, value in multimetric_value.items()]
        n_mclass_dic[mi_c] = np.sum(insp_labels.flatten() == i)
        res_string_to_print += f'{mi_c} (n={n_mclass_dic[mi_c]}): ' + ', '.join(multimetric_value_str_list) + '\n'
        for key, value in multimetric_value.items():
            res_dict[f'mc_{mi_c}_{key}'] = value

    if show_res:
        print(res_string_to_print)
    res_dict['res_string'] = res_string_to_print
    res_df = pd.DataFrame(res_dict, index=[0])
    # compute pr curve
    if not class_input and return_pr:
        y_scores = output_bos_np_softmax[:, 1]
        y_true = binary_labels.flatten()
        precision_list, recall_list, f1score_list, thresholds_list = p_r_curve(y_true, y_scores)

    if return_pr:
        return acc_binary, acc_mclass, binary_multimetrics_np, mclass_multimetrics_np, precision_list, recall_list, f1score_list, thresholds_list, res_df
    else:
        return acc_binary, acc_mclass, binary_multimetrics_np, mclass_multimetrics_np, res_df

if __name__ == '__main__':

    import torch
    output_bom = torch.Tensor([[-0.2564, -0.6555, -0.6167, -0.2392, -0.7610,  0.2800,  0.3524,  0.2232,
          0.3434, -0.4025],
        [-0.3255, -0.1070, -0.8442, -0.2080, -0.0464,  0.4900,  0.3032,  0.4905,
          0.6900, -0.4807],
        [-0.4897,  0.2226, -0.7046, -0.1382, -0.2667,  0.0196, -0.2923,  0.4146,
         -0.0556,  0.3352],
        [-0.4897,  0.2226, -0.7046, -0.1382, -0.2667,  0.0196, -0.2923,  0.4146,
         -0.0556,  0.3352]])
    output_bos = torch.Tensor([[ 0.2805,  0.0315],
        [ 0.5804, -0.3333],
        [ 0.4147,  0.0458],
        [-0.1393, -0.0531]])
    true_label = torch.Tensor([1, 2, 1, 1])

    # precision
    classk = 1
    beta = 0.5
    _, pred = output_bos.topk(1, 1, True, True)
    pred = pred.t()
    # compute predicted, true, true positive, false positive,
    # false negative, true negative of classk
    pred_classk = pred.eq(classk)
    true_classk = true_label.eq(classk)
    tp_classk = pred_classk*true_classk
    fp_classk = pred_classk*(~true_classk)
    fn_classk = (~pred_classk)*true_classk
    tn_classk = (~pred_classk)*(~true_classk)
    # compute number of predicted, true, true positive, false positive,
    # false negative, true negative of classk
    npred_classk = pred_classk.view(-1).float().sum(0)
    ntrue_classk = true_classk.view(-1).float().sum(0)
    ntp_classk = tp_classk.view(-1).float().sum(0)
    # compute precision and recall
    precision = ntp_classk/npred_classk
    recall = ntp_classk/(ntrue_classk)
    f1_score = (1+beta**2)*precision*recall/(beta**2*precision+recall)
    # print(f'precision@class={classk}: {precision}')
    # print(f'recall@class={classk}: {recall}')
