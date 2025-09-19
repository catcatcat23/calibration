import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os
from dataloader.image_resampler_pins import mp_weighted_resampler, ImageLoader2, stratified_train_val_split, \
    generate_test_pairs, LUT
from dataloader.image_resampler import DiscreteRotate
from torchvision import transforms
from utils.metrics import *
from utils.utilities import adjust_learning_rate, AverageMeter, save_model
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from special_supply_data import special_csvs_name
from util import *
from train_data import get_train_csv
from algo_test_utils import *
from models.Calibrate import *
import json
from tqdm.auto import tqdm




from typing import Optional, Tuple, Callable

def select_ece_methods(
    ece_calcu_type: str,
    *,
    n_bins: int = 15,
    kde_grid_points: int = 201,
    kde_bandwidth: Optional[float] = None,
    kde_reflect: bool = True
) -> Tuple[Optional[Callable], Optional[Callable], Optional[Callable], str, KDEer | ECEer | None]:
    """
    返回: (multi_ECEor, classwise_ECEor, overall_ECEor, classwise_tag, backend_obj)
      - classwise_tag: 'pred'（按被预测类）/'true'（按真实类）/'none'
      - backend_obj: KDEer 或 ECEer 实例（有时需要它再调用别的接口）
    """
    t = ece_calcu_type.upper()

    # ---------- KDE 系列 ----------
    if t == 'KDE':
        kde = KDEer(kde_grid_points=kde_grid_points, kde_bandwidth=kde_bandwidth, kde_reflect=kde_reflect)
        return (None, None, kde.calculate_overall_ece_kde, 'none', kde)

    if t == 'CLASSWISE_KDE':
        kde = KDEer(kde_grid_points=kde_grid_points, kde_bandwidth=kde_bandwidth, kde_reflect=kde_reflect)
        return (None, kde.calculate_classwise_ece_kde, kde.calculate_overall_ece_kde, 'pred', kde)

    if t == 'TRUEKDE':
        kde = KDEer(kde_grid_points=kde_grid_points, kde_bandwidth=kde_bandwidth, kde_reflect=kde_reflect)
        return (None, kde.calculate_trueclass_ece_kde, kde.calculate_overall_ece_kde, 'true', kde)

    if t == 'MULTI_KDE':
        kde = KDEer(kde_grid_points=kde_grid_points, kde_bandwidth=kde_bandwidth, kde_reflect=kde_reflect)
        return (kde.calculate_multiclass_ece_mce_kde, None, kde.calculate_overall_ece_kde, 'none', kde)

    # ---------- 分箱系列 ----------'ECE', 'Ada_ECE', 'Multi_ECE', 'Multi_Ada_ECE', 'Classwise_ECE', 'Classwise_Ada_ECE'
    ecer = ECEer(n_bins=n_bins)

    if t in ('ECE', 'Multi_ECE'):

        return (ecer.calculate_multiclass_ece_mce, ecer.calculate_classwise_ece_mce, None, 'none', ecer)

    # if t in ('CLASSWISE_ECE', 'CLASSWISE_ADA_ECE'):

    #     return (None, ecer.calculate_classwise_ece_mce, getattr(ecer, 'calculate_overall_ece', None), 'pred', ecer)

    # if t in ('MULTI_ECE', 'MULTI_ADA_ECE'):

    #     return (ecer.calculate_multiclass_ece_mce, None, getattr(ecer, 'calculate_overall_ece', None), 'none', ecer)

    raise ValueError(f'未知 ECE_calcu_type={ece_calcu_type}')

def process_T_value(T_value, ECE_calcu_type, output_bos_th_T1, output_bom_th_T1, bos_labels, bom_labels, multi_class_weight, ok_w, ng_w,
                     bos_ECE_results_save_path, bom_ECE_results_save_path, multi_ECEor, special_ECE_metrictor,
                     defect_decode_bos_dict, defect_decode):
    # print(f"T = {T_value}")
    output_bos_th = output_bos_th_T1 / T_value
    output_bom_th = output_bom_th_T1 / T_value

    cur_bos_ECE_results_save_path = os.path.join(bos_ECE_results_save_path, ECE_calcu_type, f'T_{T_value:.3f}')
    cur_bom_ECE_results_save_path = os.path.join(bom_ECE_results_save_path, ECE_calcu_type, f'T_{T_value:.3f}')

    os.makedirs(cur_bos_ECE_results_save_path, exist_ok=True)
    os.makedirs(cur_bom_ECE_results_save_path, exist_ok=True)

    multi_bos_ece_dict, _ = multi_ECEor(output_bos_th, bos_labels, cur_bos_ECE_results_save_path, True, True, defect_decode_bos_dict)
    multi_bom_ece_dict, _ = multi_ECEor(output_bom_th, bom_labels, cur_bom_ECE_results_save_path, True, True, defect_decode)

    classwise_bos_ece_dict, _ = special_ECE_metrictor(output_bos_th, bos_labels, cur_bos_ECE_results_save_path, True, True, defect_decode_bos_dict)
    classwise_bom_ece_dict, _ = special_ECE_metrictor(output_bom_th, bom_labels, cur_bom_ECE_results_save_path, True, True, defect_decode)

    cur_T_classwise_bom_ece = sum(multi_class_weight[d] * e for d, e in classwise_bom_ece_dict.items())
    cur_T_classwise_bos_ece = ok_w * classwise_bos_ece_dict['ok'] + ng_w * classwise_bos_ece_dict['ng']

    cur_T_total_bos_ece = cur_T_classwise_bos_ece + multi_bos_ece_dict['multiclass_ece']
    cur_T_total_bom_ece = cur_T_classwise_bom_ece + multi_bom_ece_dict['multiclass_ece']

    return T_value, cur_T_total_bos_ece, cur_T_total_bom_ece, cur_T_classwise_bom_ece, cur_T_classwise_bos_ece, multi_bos_ece_dict, multi_bom_ece_dict

def find_max_valid_threshold(output_bos_th, bos_labels, ng_thres_dt = 0.01, default_thres = 0.8):
    output_bos_th_softmax = F.softmax(output_bos_th, dim=1)
    confidences, predictions = torch.max(output_bos_th_softmax, 1)

    ok_labels = bos_labels == 0
    ng_labels = bos_labels == 1

    ok_confidences = confidences[ok_labels]
    ng_confidences = confidences[ng_labels]
    
    ok_predictions = predictions[ok_labels]
    ng_predictions = predictions[ng_labels]

    label_ok_prediction_ng = ok_predictions == 1
    label_ok_prediction_ok = ok_predictions == 0

    ok_false_conf = ok_confidences[label_ok_prediction_ng] # ok预测为ng的置信度
    ok_recall = ok_confidences[label_ok_prediction_ok] # ok置信度大于0.5

    label_ng_prediction_ok = ng_predictions == 0
    label_ng_prediction_ng = ng_predictions == 1

    ng_omit_conf = ng_confidences[label_ng_prediction_ok] # ng 预测为ok的置信度，所以这些类的ng 置信度必然低于0.5了
    ng_recall_conf = ng_confidences[label_ng_prediction_ng]

    total_ok_false = ok_false_conf.shape[0]
    total_ng_omit = ng_omit_conf.shape[0]

    # ng_threshold = np.arange(0.5, 1 + ng_thres_dt, ng_thres_dt, atol=1e-9) # 这个会存在精度问题
    ng_threshold = np.linspace(0.5, 1, int((1 - 0.5) / ng_thres_dt) + 1)
    if not np.any(np.isclose(ng_threshold, default_thres)):
        ng_threshold = np.sort(np.append(ng_threshold, default_thres))
        
    # 批量计算阈值条件的布尔掩码
    ok_correct_mask = ok_false_conf.unsqueeze(1) < torch.tensor(ng_threshold, device=ok_false_conf.device).float().unsqueeze(0)
    ng_false_correct_mask = ng_recall_conf.unsqueeze(1) < torch.tensor(ng_threshold, device = ng_recall_conf.device).float().unsqueeze(0)
    # 计算每个阈值条件下的正确和错误数量
    ok_correct_nums = ok_correct_mask.sum(dim=0)
    ng_false_correct_nums = ng_false_correct_mask.sum(dim=0)

    # 计算绝对正确数和剩余错误数
    abs_valid_correct_nums = ok_correct_nums - ng_false_correct_nums
    remain_wrong_nums = total_ok_false - ok_correct_nums + total_ng_omit + ng_false_correct_nums

    # 获取最大正确数和最小错误数及对应阈值
    total_max_correct_nums = abs_valid_correct_nums.max().item()
    total_max_correct_nums_indices = torch.where(abs_valid_correct_nums == total_max_correct_nums)[0]
    total_max_correct_nums_thresholds = ng_threshold[total_max_correct_nums_indices[-1].item()]

    total_min_wrong_nums = remain_wrong_nums.min().item()
    total_min_wrong_nums_indices = torch.where(remain_wrong_nums == total_min_wrong_nums)[0]
    total_min_wrong_nums_thresholds = ng_threshold[total_min_wrong_nums_indices[-1].item()]
    # min_wrong_nums_thresholds = ng_threshold[remain_wrong_nums.argmin()]

    # 获取默认阈值下的正确和错误数
    default_correct_nums = abs_valid_correct_nums[ng_threshold == default_thres].item()
    default_wrong_nums = remain_wrong_nums[ng_threshold == default_thres].item()

    # ok_max_correct_nums = ok_correct_nums.max().item() 肯定是最大阈值的时候，矫正数量最大，单独计算矫正数量没有意义
    # ng_max_false_correct_nums = ng_false_correct_nums.max().item() # 当前设置的阈值是ng的判断阈值，所以从0.5开始，只会有越来越多的漏报，计算最大漏报数量没有意义

    # ok_max_correct_indices = torch.where(ok_correct_nums == ok_max_correct_nums)[0]
    # ok_max_correct_thresholds = ng_threshold[ok_max_correct_indices[-1].item()]
    # ng_false_match_ok_max_correct = ng_false_correct_nums[ok_max_correct_indices[-1].item()]
    # ng_max_false_correct_indices = torch.where(ng_false_correct_nums == ng_max_false_correct_nums)[0]   

    # for cur_ng_threshold in ng_threshold:
    #     ok_correct_nums = (ok_false_conf < cur_ng_threshold).sum()
    #     ng_false_correct_nums = (ng_recall_conf < cur_ng_threshold).sum()

    #     abs_valid_correct_nums = ok_correct_nums - ng_false_correct_nums
    #     remain_wrong_nums = total_ok_false - ok_correct_nums + total_ng_omit + ng_false_correct_nums

    #     if abs_valid_correct_nums >= max_correct_nums:
    #         max_correct_nums = abs_valid_correct_nums
    #         max_correct_nums_thresholds = cur_ng_threshold
        
    #     if remain_wrong_nums <= min_wrong_nums:
    #         min_wrong_nums = remain_wrong_nums
    #         min_wrong_nums_thresholds = cur_ng_threshold

    #     if cur_ng_threshold == default_thres:
    #         default_correct_nums = abs_valid_correct_nums
    #         default_wrong_nums = remain_wrong_nums
    
    
    return total_max_correct_nums, total_min_wrong_nums, total_max_correct_nums_thresholds, total_min_wrong_nums_thresholds, \
            default_correct_nums, default_wrong_nums

def get_p_r_list(output_bos_all, output_bom_all, label_binary_all, label_mclass_all):
    output_bos_np = output_bos_all.detach().cpu().numpy()
    output_bom_np = output_bom_all.detach().cpu().numpy()
    binary_labels = label_binary_all.detach().cpu().numpy()
    insp_labels = label_mclass_all.detach().cpu().numpy()

    output_bos_np_softmax = softmax(output_bos_np, axis=1)
    output_bom_np_softmax = softmax(output_bom_np, axis=1)

    y_scores = output_bos_np_softmax[:, 1]
    y_true = binary_labels.flatten()
    precision_list, recall_list, f1score_list, thresholds_list = p_r_curve(y_true, y_scores)

    return precision_list, recall_list, f1score_list, thresholds_list

def plot_p_r(precision_list, recall_list, f1score_list, thresholds_list, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(4, 6))
    axes[0].plot(recall_list, precision_list, 'r-')
    # axes[0].set_title(f'PR curve')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precesion')

    axes[1].plot(thresholds_list, f1score_list, 'b-')
    # axes[1].set_title(f'F1-Thre curve')
    max_f1 = np.max(f1score_list)
    max_f1_thre = thresholds_list[np.argmax(f1score_list)]
    axes[1].plot(max_f1_thre, max_f1, 'ro', label=f'f1={max_f1:.3f} \nth={max_f1_thre:.3f}')
    axes[1].legend(loc=2)
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('F1 score')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cur_T_p_r(calibrate_models, p_r_dir, precision_list, recall_list, f1score_list, thresholds_list):
    T_value = calibrate_models.T.item()
    save_path = os.path.join(p_r_dir, f'temprature_{T_value}.png')
    plot_p_r(precision_list, recall_list, f1score_list, thresholds_list, save_path)

def train(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists, use_amp=False, scaler=None, kd_config=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()
    model.pre_fintune() # 冻结原始模型权重
    if kd_config is not None:
        kd_ratio = kd_config['kd_ratio']
        teacher_model = kd_config['teacher_model']
        teacher_model.eval()
        kd_type = kd_config['kd_type']
        T = kd_config['temperature']
    # label1是ref_y, label2是insp_y, binary_y = binary_y
    for batch_id, (img1, img2, label1, label2, binary_y) in enumerate(train_loader):  #, position
#         print(img1.shape, img2.shape, type(label1), type(label2), type(binary_y), type(position))
        if gpu_exists:
            img1, img2, label1, label2, binary_y = Variable(img1.cuda()), Variable(img2.cuda()), Variable(
                label1.cuda()), Variable(label2.cuda()), Variable(binary_y.cuda())
        else:
            img1, img2, label1, label2, binary_y = Variable(img1), Variable(img2), Variable(label1), Variable(
                label2), Variable(binary_y)

        label_binary = binary_y.type(torch.int64)
        # if use mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # compute loss and accuracy
            # output_bos, output_bom = model(img1, img2)
            output_bos, output_bom, feature_1, feature_2 = model(img1, img2)

            loss_contrastive = cl_loss(feature_1, feature_2, label_binary) 
            loss_mclass = criterion_bom(output_bom, label2)

            if output_bos is not None:
                # label_binary = (label1 == label2).type(torch.float32).reshape([-1,1])
                loss_binary = criterion_bos(output_bos, label_binary) * output_bom.shape[1]
                loss = (loss_binary + loss_mclass) / 2
                if (batch_id > 0) and (batch_id % (len(train_loader) - 1) == 0):
                        print(f'binary_loss={loss_binary}, mclass_loss={loss_mclass}， n_class={output_bom.shape}, contrastive loss={loss_contrastive}')
            else:
                loss = loss_mclass
            
            loss = 4 * loss + loss_contrastive
            if kd_config is not None:
                # only applicable for dual2 for now
                with torch.no_grad():
                    teacher_output_bos, teacher_output_bom = teacher_model(img1, img2)
                teacher_soft_targets_bom = F.softmax(teacher_output_bom / T, dim=-1)
                if output_bos is not None:
                    teacher_soft_targets_bos = F.softmax(teacher_output_bos / T, dim=-1)

                if kd_type == "ce":
                    soft_output_bom = F.log_softmax(output_bom / T, dim=-1)
                    kd_loss_bom = torch.mean(-torch.sum(teacher_soft_targets_bom * soft_output_bom, 1)) * (T**2)
                    if output_bos is not None:
                        soft_output_bos = F.log_softmax(output_bos / T, dim=-1)
                        kd_loss_bos = torch.mean(-torch.sum(teacher_soft_targets_bos * soft_output_bos, 1)) * (T**2)
                else:
                    kd_loss_bom = F.mse_loss(output_bom, teacher_soft_targets_bom)
                    if output_bos is not None:
                        kd_loss_bos = F.mse_loss(output_bos, teacher_soft_targets_bos)

                # compute the overall loss
                if output_bos is not None:
                    loss += kd_ratio*(kd_loss_bos * output_bom.shape[1] + kd_loss_bom)
                    if (batch_id > 0) and (batch_id % (len(train_loader) - 1) == 0):
                        print(f'kd loss bos={kd_loss_bos}, kd_loss_bom={kd_loss_bom}， kd_ratio={kd_ratio}')
                else:
                    loss += kd_ratio*kd_loss_bom

        # compute gradient and perform gradient update
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()

            if loss > 1000:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if loss > 1000:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # update record
        acc1 = accuracy(output_bom, label2, topk=(1,))
        batch_size = img1.size(0)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc1[0], batch_size)

    return losses.avg, accuracies.avg

def train_no_cl(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists, use_amp=False, scaler=None, kd_config=None):
    
    
    """
    说明：
    - model(img1, img2) 统一返回“像 logits”的张量：
        * TS: logits/T
        * ETS: log(q) （CrossEntropy/Focal 依然正确）
    - criterion_bom / criterion_bos 可以是 CrossEntropyLoss 或支持 logits 的 FocalLoss
    - KD:
        * kd_type == "ce" : KL(teacher || student) 的稳定写法（teacher prob × (log teacher - log student)）
        * kd_type == "mse": 对概率做 MSE（student 用 softmax(output)；ETS 时 softmax(log(q))=q）
    """ 
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()
    if kd_config is not None:
        kd_ratio = kd_config['kd_ratio']
        teacher_model = kd_config['teacher_model']
        teacher_model.eval()
        kd_type = kd_config['kd_type']
        T = kd_config['temperature']
        eps = 1e-8
    # label1是ref_y, label2是insp_y, binary_y = binary_y
    for batch_id, (img1, img2, label1, label2, binary_y) in enumerate(train_loader):  #, position
#         print(img1.shape, img2.shape, type(label1), type(label2), type(binary_y), type(position))
        if gpu_exists:
            img1, img2, label1, label2, binary_y = Variable(img1.cuda()), Variable(img2.cuda()), Variable(
                label1.cuda()), Variable(label2.cuda()), Variable(binary_y.cuda())
        else:
            img1, img2, label1, label2, binary_y = Variable(img1), Variable(img2), Variable(label1), Variable(
                label2), Variable(binary_y)

        label_binary = binary_y.type(torch.int64)
        # if use mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # compute loss and accuracy
            output_bos, output_bom = model(img1, img2)

            loss_mclass = criterion_bom(output_bom, label2)

            if output_bos is not None:
                # label_binary = (label1 == label2).type(torch.float32).reshape([-1,1])
                loss_binary = criterion_bos(output_bos, label_binary) * output_bom.shape[1]
                loss = (loss_binary + loss_mclass) / 2
                if (batch_id > 0) and (batch_id % (len(train_loader) - 1) == 0):
                        print(f'binary_loss={loss_binary}, mclass_loss={loss_mclass}， n_class={output_bom.shape}')
            else:
                loss = loss_mclass
            
            if kd_config is not None:
                # only applicable for dual2 for now
                with torch.no_grad():
                    teacher_output_bos, teacher_output_bom = teacher_model(img1, img2)
                teacher_soft_targets_bom = F.softmax(teacher_output_bom / T, dim=-1)
                if output_bos is not None:
                    teacher_soft_targets_bos = F.softmax(teacher_output_bos / T, dim=-1)

                if kd_type == "ce":
                    soft_output_bom = F.log_softmax(output_bom / T, dim=-1)
                    kd_loss_bom = torch.mean(-torch.sum(teacher_soft_targets_bom * soft_output_bom, 1)) * (T**2)
                    if output_bos is not None:
                        soft_output_bos = F.log_softmax(output_bos / T, dim=-1)
                        kd_loss_bos = torch.mean(-torch.sum(teacher_soft_targets_bos * soft_output_bos, 1)) * (T**2)
                else:
                    kd_loss_bom = F.mse_loss(output_bom, teacher_soft_targets_bom)
                    if output_bos is not None:
                        kd_loss_bos = F.mse_loss(output_bos, teacher_soft_targets_bos)

                # compute the overall loss
                if output_bos is not None:
                    loss += kd_ratio*(kd_loss_bos * output_bom.shape[1] + kd_loss_bom)
                    if (batch_id > 0) and (batch_id % (len(train_loader) - 1) == 0):
                        print(f'kd loss bos={kd_loss_bos}, kd_loss_bom={kd_loss_bom}， kd_ratio={kd_ratio}')
                else:
                    loss += kd_ratio*kd_loss_bom

        # compute gradient and perform gradient update
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()

            if loss > 1000:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if loss > 1000:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # update record
        acc1 = accuracy(output_bom, label2, topk=(1,))
        batch_size = img1.size(0)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc1[0], batch_size)

    return losses.avg, accuracies.avg

def valer(val_loader, output_type, model, criterion_bos, criterion_bom, gpu_exists, visualize, use_amp, selection_score):
    if "CL" in output_type:
        val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions, val_imgs, \
        precision_list, recall_list, f1score_list, thresholds_list, output_bos_th, output_bom_th, bos_labels, bom_labels = val(
            val_loader, model, criterion_bos, criterion_bom, gpu_exists, visualize=True, use_amp=use_amp, selection_score=selection_score)

    else:
        val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions, val_imgs, \
        precision_list, recall_list, f1score_list, thresholds_list, output_bos_th, output_bom_th, bos_labels, bom_labels  = val_no_cl(
            val_loader, model, criterion_bos, criterion_bom, gpu_exists, visualize=True, use_amp=use_amp, selection_score=selection_score)
    
    return val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions, val_imgs, \
        precision_list, recall_list, f1score_list, thresholds_list, output_bos_th, output_bom_th, bos_labels, bom_labels

def val(val_loader, model, criterion_bos, criterion_bom, gpu_exists, visualize=False, use_amp=False, selection_score='recall'):

    if selection_score == 'recall':
        metric_idx = 1
    elif selection_score == 'f1':
        metric_idx = 2

    losses = AverageMeter()
    accuracies_mclass = AverageMeter()
    accuracies_binary = AverageMeter()
    if visualize:
        val_imgs = []

    # switch to evaluate mode
    model.eval()
    val_label_positions = []
    output_bos_list = []
    label_binary_list = []
    output_bom_list = []
    label_mclass_list = []

    for batch_id, (img1, img2, label1, label2, binary_y) in enumerate(val_loader): #, position

        if gpu_exists:
            img1, img2, label1, label2, binary_y = Variable(img1.cuda()), Variable(img2.cuda()), Variable(
                label1.cuda()), Variable(label2.cuda()), Variable(binary_y.cuda())
        else:
            img1, img2, label1, label2, binary_y = Variable(img1), Variable(img2), Variable(label1), Variable(
                label2), Variable(binary_y)

        label_binary = binary_y.type(torch.int64)

        # compute loss and accuracy
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                # output_bos, output_bom = model(img1, img2)
                output_bos, output_bom, _, _ = model(img1, img2)
                # label_binary = (label1 == label2).type(torch.float32).reshape([-1,1])
                if output_bos is not None:
                    loss = (criterion_bos(output_bos, label_binary) + criterion_bom(output_bom, label2)) / 2
                else:
                    loss = criterion_bom(output_bom, label2)

        batch_size = img1.size(0)
        if output_bos is not None:
            acc_binary = accuracy(output_bos, label_binary, topk=(1,))
            accuracies_binary.update(acc_binary[0], batch_size)
            output_bos_list.append(output_bos)
            output_bos_np = output_bos.detach().cpu().numpy()
        else:
            output_bos_np = 0*output_bom.detach().cpu().numpy()

        acc_mclass = accuracy(output_bom, label2, topk=(1,))
        # update record
        losses.update(loss.item(), batch_size)
        accuracies_mclass.update(acc_mclass[0], batch_size)
        label_binary_list.append(label_binary)
        output_bom_list.append(output_bom)
        label_mclass_list.append(label2)

        # if visualize:
        #     val_imgs.append([img1.cpu().numpy(), img2.cpu().numpy()])
        #     val_label_positions.append([output_bos_np, output_bom.detach().cpu().numpy(),
        #                                 label1, label2, label_binary, position])
        # else:
        #     val_label_positions.append([label1, label2, label_binary, position])

    if output_bos is not None:
        label_binary_all = torch.cat(label_binary_list, dim=0)
        output_bos_all = torch.cat(output_bos_list, dim=0)
        recall_bi = classk_metric(output_bos_all, label_binary_all, 1, input_type='torch')[metric_idx]
    else:
        recall_bi = 0

    output_bom_all = torch.cat(output_bom_list, dim=0)
    label_mclass_all = torch.cat(label_mclass_list, dim=0)
    # print(output_bos_all, label_binary_all)
    recall_mclass = classk_metric(output_bom_all, label_mclass_all, -1, input_type='torch')[metric_idx]

    precision_list, recall_list, f1score_list, thresholds_list = get_p_r_list(output_bos_all, output_bom_all, label_binary_all, label_mclass_all)

    output_bos_th = output_bos_all.detach().clone()
    output_bom_th = output_bom_all.detach().clone()
    bos_labels = label_binary_all.detach().clone()
    bom_labels = label_mclass_all.detach().clone()

    if visualize:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions, \
                val_imgs, precision_list, recall_list, f1score_list, thresholds_list, output_bos_th, output_bom_th, bos_labels, bom_labels
    else:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions, \
                precision_list, recall_list, f1score_list, thresholds_list, output_bos_th, output_bom_th, bos_labels, bom_labels
    
def val_no_cl(val_loader, model, criterion_bos, criterion_bom, gpu_exists,
              visualize=False, use_amp=False, selection_score='recall'):

    if selection_score == 'recall':
        metric_idx = 1
    elif selection_score == 'f1':
        metric_idx = 2

    losses = AverageMeter()
    accuracies_mclass = AverageMeter()
    accuracies_binary = AverageMeter()
    if visualize:
        val_imgs = []

    model.eval()
    val_label_positions = []
    output_bos_list, label_binary_list = [], []
    output_bom_list, label_mclass_list = [], []

    device_type = 'cuda'
    amp_enabled = (use_amp and gpu_exists)

    with torch.no_grad():
        for batch_id, (img1, img2, label1, label2, binary_y) in enumerate(val_loader):
            if gpu_exists:
                img1 = torch.FloatTensor(img1).cuda()
                img2 = torch.FloatTensor(img2).cuda()
                label1 = torch.FloatTensor(label1).cuda()
                label2 = torch.FloatTensor(label2).cuda()
                binary_y = torch.FloatTensor(binary_y).cuda()
            else:
                img1 = torch.FloatTensor(img1)
                img2 = torch.FloatTensor(img2)
                label1 = torch.FloatTensor(label1)
                label2 = torch.FloatTensor(label2)
                binary_y = torch.FloatTensor(binary_y)

            label_binary = binary_y.type(torch.int64)
            label2 = label2.type(torch.int64)

            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
                output_bos, output_bom = model(img1, img2)
                if output_bos is not None:
                    loss = (criterion_bos(output_bos, label_binary) + criterion_bom(output_bom, label2)) / 2
                else:
                    loss = criterion_bom(output_bom, label2)

            batch_size = img1.size(0)
            losses.update(loss.item(), batch_size)

            # 多分类准确率
            acc_mclass = accuracy(output_bom, label2, topk=(1,))
            accuracies_mclass.update(acc_mclass[0], batch_size)
            output_bom_list.append(output_bom)
            label_mclass_list.append(label2)

            # 二分类（BOS）准确率（仅当存在 BOS 输出）
            if output_bos is not None:
                acc_binary = accuracy(output_bos, label_binary, topk=(1,))
                accuracies_binary.update(acc_binary[0], batch_size)
                output_bos_list.append(output_bos)
                label_binary_list.append(label_binary)

    # -------- 聚合阶段：全部加保护 --------
    has_bos = (len(output_bos_list) > 0)

    # 多分类聚合（保证非空再 cat）
    if len(output_bom_list) > 0:
        output_bom_all = torch.cat(output_bom_list, dim=0)
        label_mclass_all = torch.cat(label_mclass_list, dim=0)
    else:
        # 空验证集的兜底（尽量返回空张量/None，避免后续再炸）
        output_bom_all = None
        label_mclass_all = None

    # 二分类（BOS）聚合
    if has_bos:
        label_binary_all = torch.cat(label_binary_list, dim=0)
        output_bos_all = torch.cat(output_bos_list, dim=0)
        recall_bi = classk_metric(output_bos_all, label_binary_all, 1, input_type='torch')[metric_idx]
    else:
        label_binary_all = None
        output_bos_all = None
        recall_bi = 0

    # PR/F1 列表：只有在 BOS 存在时才计算二分类相关；否则给空列表即可
    if (output_bom_all is not None) and has_bos:
        precision_list, recall_list, f1score_list, thresholds_list = \
            get_p_r_list(output_bos_all, output_bom_all, label_binary_all, label_mclass_all)
    else:
        precision_list = recall_list = f1score_list = thresholds_list = []

    # 准备返回的“快照”张量（没有 BOS 就返回 None）
    output_bos_th = (output_bos_all.detach().clone() if has_bos else None)
    bos_labels   = (label_binary_all.detach().clone() if has_bos else None)

    if output_bom_all is not None:
        output_bom_th = output_bom_all.detach().clone()
        bom_labels    = label_mclass_all.detach().clone()
    else:
        output_bom_th = None
        bom_labels    = None

    if visualize:
        return (losses.avg, accuracies_binary.avg, accuracies_mclass.avg,
                recall_bi, classk_metric(output_bom_all, label_mclass_all, -1, input_type='torch')[metric_idx] if output_bom_all is not None else 0,
                val_label_positions, val_imgs,
                precision_list, recall_list, f1score_list, thresholds_list,
                output_bos_th, output_bom_th, bos_labels, bom_labels)
    else:
        return (losses.avg, accuracies_binary.avg, accuracies_mclass.avg,
                recall_bi, classk_metric(output_bom_all, label_mclass_all, -1, input_type='torch')[metric_idx] if output_bom_all is not None else 0,
                val_label_positions,
                precision_list, recall_list, f1score_list, thresholds_list,
                output_bos_th, output_bom_th, bos_labels, bom_labels)


def load_calibrate_model(args,  model):
    print(f'Loading checkpoint {args.resume}')
    checkpoint = torch.load(args.resume)
    loaded_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()

    if args.reload_mode == 'full':
        model.load_state_dict(loaded_state_dict)
    elif args.reload_mode == 'backbone':
        for k in loaded_state_dict:
            if ('cnn_encoder' in k) and (k in model_state_dict):
                model_state_dict[k] = loaded_state_dict[k]
        model.load_state_dict(model_state_dict)
    elif args.reload_mode == 'skip_mismatch':
        for k in model_state_dict:
            if (k in loaded_state_dict) and (loaded_state_dict[k].shape == model_state_dict[k].shape):
                model_state_dict[k] = loaded_state_dict[k]
            else:
                print(f"Skip parameter: {k}")
        model.load_state_dict(model_state_dict)
    else:
        assert False

    print('Fininsh loading checkpoint')

    calibrate_models = Calibrate_Model(model)
    return calibrate_models

def load_resume_model(backbone_arch, output_type, pretrained, n_class, n_units):
    # define classifier model
    if 'resnetsp' in backbone_arch and 'CL' not in output_type:
        from models.MPB3 import MPB3net
        model = MPB3net(backbone=backbone_arch, pretrained=pretrained, n_class=n_class, n_units=n_units, output_form=output_type)
    elif 'CL' in output_type:
        from models.MPB3_ConstLearning import MPB3net
        model = MPB3net(backbone=backbone_arch, pretrained=pretrained, n_class=n_class, n_units=n_units, output_form=output_type)

    else:
        from models.MPB3 import MPB3net
        model = MPB3net(backbone=backbone_arch, pretrained=pretrained, n_class=n_class, n_units=n_units, output_form=output_type)

    n_params = sum([p.numel() for p in model.parameters()])
    print(f'model has {n_params / (1024 * 1024)} M params')

    return model

def set_fintune_param(optimizer_type, calibrate_models, init_lr, weight_decay, lr_schedule, use_amp, lr_step_size, lr_gamma):
    
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(calibrate_models.parameters(), lr=init_lr, momentum=0.9, weight_decay=weight_decay)

    elif optimizer_type == "sgdft":
        output_layers = [param for param in calibrate_models.model.head_bos.parameters()] + [param for param in
                                                                            calibrate_models.model.head_bom.parameters()]
        optimizer = torch.optim.SGD([{'params': calibrate_models.model.cnn_encoder.parameters(), 'lr': init_lr / 10},
                                    {'params': calibrate_models.T, 'lr': init_lr},
                                    {'params': output_layers},
                                    ],
                                    lr=init_lr, momentum=0.9, weight_decay=weight_decay)

    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(calibrate_models.model.parameters(), lr=init_lr, momentum=0.9,
                                        weight_decay=weight_decay, eps=0.0316, alpha=0.9)

    if lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.lr_schedule_T, eta_min=0.00001)

    elif lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        scaler = None

    return optimizer, scheduler, scaler

def main(args):
    date = args.date
    # set data path and set-up configs
    img_folder = args.data
    ckp_path = args.ckp
    region = args.region
    smooth = args.smooth
    rs_img_size_h = args.resize_h
    rs_img_size_w = args.resize_w
    seed = args.seed
    selection_score = args.score
    k_factor = args.k
    val_ratio = args.val_ratio
    batch_size = args.batch_size
    n_workers = args.worker
    gamma = args.gamma
    init_lr = args.learning_rate
    start_epoch = args.start_epoch
    output_type = args.output_type
    n_unit_binary = args.n_unit_binary
    n_unit_mclass =args.n_unit_mclass
    n_units = [n_unit_binary, n_unit_mclass]
    epochs = args.epochs
    gpu_id = args.gpu
    lut_p = args.lut_p
    save_checkpoint = True
    optimizer_type = args.optimizer_type
    lr_schedule = args.lr_schedule
    lr_step_size = args.lr_step_size
    lr_gamma = args.lr_gamma
    jitter = args.jitter
    weight_decay = args.weight_decay
    version_name = args.version_name
    # kd arguments
    kd_ratio = args.kd_ratio
    kd_T = args.kd_T
    kd_type = args.kd_type
    label_confidence = args.label_conf
    higher_threshold = args.higher_threshold
    img_folder_mnt = f'./results/{version_name}_csv'
    annotation_folder = os.path.join(img_folder, 'merged_annotation', date)
    annotation_folder_mnt = os.path.join(img_folder_mnt, f'{region}_annotation', date)

    tb_writer = SummaryWriter(os.path.join(ckp_path, args.tb_logdir))
    os.makedirs(annotation_folder_mnt, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    # data augmentation
    transform_mean = [0.485, 0.456, 0.406]
    transform_std = [0.229, 0.224, 0.225]
    transform_separate = transforms.Compose([
                                LUT(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'), lut_p=lut_p),
                                transforms.Resize((int(rs_img_size_h / 0.95), int(rs_img_size_w / 0.95))),
                                transforms.RandomCrop((rs_img_size_h, rs_img_size_w)),
                                transforms.ColorJitter(
                                        brightness=0.1, contrast=0.1, saturation=0.15, hue=0.1
                                    )])
    transform_same = transforms.Compose([
        LUT(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'), lut_p=lut_p),
        transforms.Resize((int(rs_img_size_h + 3), int(rs_img_size_w + 5))),
                                        transforms.RandomCrop((rs_img_size_h, rs_img_size_w)),
                                        transforms.ColorJitter(
                                            brightness=0.1, contrast=0.1, saturation=0.15, hue=0.1
                                        )
                                    ])
    transform_sync = transforms.Compose([
                                        DiscreteRotate(angles=[0, 180]),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ColorJitter(
                                            brightness=jitter, contrast=jitter, saturation=jitter, hue=0.2
                                        ),
                                        transforms.Normalize(mean=transform_mean,
                                                             std=transform_std)
                                    ])
    # decode the defect id back to defect label
    if region == 'singlepinpad':
        n_max_pairs_val = 50
        n_max_pairs_train = 50

        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_code_val = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        annotation_filename, val_annotation_filename, test_annotation_filenames, aug_train_pair_data_filenames, aug_val_pair_data_filenames = \
            get_train_csv(annotation_folder, region, version_name, args.special_data)
        
    elif region == 'padgroup':
        n_max_pairs_val = 15
        n_max_pairs_train = 20
        if 'slim' in version_name:
            defect_code = {'ok': 0, 'solder_shortage': 7}
            defect_code_val = {'ok': 0, 'solder_shortage': 7}
        else:
            defect_code = {'ok': 0, 'missing': 1, 'solder_shortage': 7}
            defect_code_val = {'ok': 0, 'missing': 1, 'solder_shortage': 7}

        annotation_filename, val_annotation_filename, test_annotation_filenames, aug_train_pair_data_filenames, aug_val_pair_data_filenames = \
            get_train_csv(annotation_folder, region, version_name, args.special_data)
        
        # val_image_pair_path_list = get_test_all_csvs(img_folder, date,  'padgroup', args.valdataset_target)
  
    elif region == 'singlepad':
        transform_separate = transforms.Compose([
        LUT(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'), lut_p=lut_p),
            transforms.Resize((int(rs_img_size_h / 0.95), int(rs_img_size_w / 0.95))),
            transforms.RandomCrop((rs_img_size_h, rs_img_size_w)),
            # DiscreteRotate(angles=[0, 90, 180, 270]),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.15, hue=0.1
            )])
        transform_same = transforms.Compose([ # 非同步的
        LUT(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'), lut_p=lut_p),
            transforms.Resize((int(rs_img_size_h + 3), int(rs_img_size_w + 3))),
            transforms.RandomCrop((rs_img_size_h, rs_img_size_w)),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.15, hue=0.1
            )
        ])  
        transform_sync = transforms.Compose([  # 同步的
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            DiscreteRotate(angles=[0, 90, 180, 270]),
            # transforms.GaussianBlur()
            transforms.ColorJitter(
                brightness=jitter, contrast=jitter, saturation=jitter, hue=0.2
            ),
            transforms.Normalize(mean=transform_mean,
                                 std=transform_std)
        ])
        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_code_val = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}

        n_max_pairs_val = 15
        n_max_pairs_train = 20
        # annotation_filename = os.path.join(annotation_folder, f'train_labels_{region}.csv')
        # val_annotation_filename = os.path.join(annotation_folder, f'val_labels_{region}.csv')
        # annotation_filename, val_annotation_filename, test_annotation_filenames, aug_train_pair_data_filenames, aug_val_pair_data_filenames = \
        #     get_train_csv(annotation_folder, region, version_name, args.special_data)
        # train_val_image_pair_path_list = aug_train_pair_data_filenames
        train_val_image_pair_path_list = get_test_all_csvs(img_folder, date,  'singlepad', args.calibrate_dataset, 'rgb')
        aug_train_pair_data_filenames = train_val_image_pair_path_list
        aug_val_pair_data_filenames = aug_train_pair_data_filenames
        test_annotation_filenames = aug_train_pair_data_filenames

        # annotation_filename = None
        # val_annotation_filename = None

    # 训练过程中根据指定测试集的指标保存模型
    if args.if_save_test_best_model:
        print(f"if_save_test_best_model: {args.if_save_test_best_model}")
        # 如果valdataset_target不是follow_val，则按照指定测试集进行训练
        if args.valdataset_target != 'follow_val':
            val_image_pair_path_list = get_test_all_csvs(img_folder, date,  region, args.valdataset_target)
        else:
            val_image_pair_path_list = aug_val_pair_data_filenames

        val_image_pair_path_list = list(set(val_image_pair_path_list))
        print(f'args.valdataset_target = {args.valdataset_target} || {val_image_pair_path_list}', )
        version_folder = os.path.basename(ckp_path)
        _, _, _, _, _, _, _, binary_labels_test_v, _, \
        insp_labels_test_v, ref_image_batches, insp_image_batches, _ = get_test_df(args, 
                                                        aug_val_pair_data_filenames, 
                                                        region, 
                                                        version_folder, 
                                                        batch_size, 
                                                        img_folder)


    n_class = len(defect_code)
    defect_code_link = {v: i for i, v in enumerate(defect_code.values())}
    # defect_decode = {v: k for k, v in defect_code_link.items()}
    defect_class_considered = list(defect_code.values())
    defect_class_considered_val = list(defect_code_val.values())
    print(f'train data csvs ={aug_train_pair_data_filenames}')
    print(f'val data csvs ={aug_val_pair_data_filenames}')
    defect_decode = {i: k for i, k in enumerate(defect_code.keys())}

    if args.amp_lvl == 'f16':
        use_amp = True
    else:
        use_amp = False

    # learning rate scheduler
    if 'wl' in version_name:
        weighted_loss = True
    else:
        weighted_loss = False

    verbose_frequency = args.verbose_frequency
    backbone_arch = args.arch.split('_')[0]
    if 'pretrained' in args.arch:
        pretrained = True
    else:
        pretrained = False
    run_mode = args.mode
    best_checkpoint_path = os.path.join(ckp_path,
                                        f'{region}{args.arch}rs{rs_img_size_w}{rs_img_size_h}s{seed}c{n_class}val{val_ratio}b{batch_size}_ckp_best{version_name}{gamma}{smooth}j{jitter}lr{init_lr}nb{n_unit_binary}nm{n_unit_mclass}{output_type}top0.pth.tar')

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"use gpu: {gpu_id} to train.")
        gpu_exists = True
    else:
        gpu_exists = False

    # fix random seed for reproducibility

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = [], [], [], []
    if aug_train_pair_data_filenames is not None:
        Xtrain_resampled_, ytrain_resampled_, ybinary_resampled_, material_train_resampled_, _ = process_csv_files(train_val_image_pair_path_list, 
                                                                                                            args, 
                                                                                                            label_confidence, 
                                                                                                            defect_class_considered, 
                                                                                                            defect_code_link, 
                                                                                                            'train',
                                                                                                            drop_dup = True
                                                                                                            )
        Xtrain_resampled += Xtrain_resampled_
        ytrain_resampled += ytrain_resampled_
        ybinary_resampled += ybinary_resampled_
        material_train_resampled += material_train_resampled_

    if args.save_traintestval_infos:
        save_traintestval_infos(Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled, annotation_folder_mnt, region)

    print(f'n_train_pairs = {len(material_train_resampled)}')

    material_train_resampled = [str(x) if not isinstance(x, str) else x for x in material_train_resampled]

    train_dataset = ImageLoader2(img_folder, Xtrain_resampled, ytrain_resampled, ybinary_resampled,
                                material_train_resampled,
                                compression_p = args.compression_p, 
                                p_range = args.p_range,
                                select_p = args.select_p,
                                transform=transform_separate,
                                transform_same=transform_same,
                                transform_sync=transform_sync)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=n_workers, pin_memory=True)


    if args.aug_img:
        sharpness_save =args.region + '-' + str(args.sharpness_factor) + '-' + str(args.sharpness_p)
        os.makedirs(sharpness_save, exist_ok=True)
        os.makedirs(sharpness_save + '-ori', exist_ok=True)
        transform = transforms.Compose([
                                        # transforms.ColorJitter(
                                        #     brightness=args.colorjitter[0], contrast=args.colorjitter[1],
                                        #     saturation=args.colorjitter[2], hue=args.colorjitter[3]
                                        # ),
                                        transforms.RandomAdjustSharpness(sharpness_factor=args.sharpness_factor, p=1)
                                     ])
            
    else:
        transform = None
        sharpness_save = None
        sharpness_p = 0
    version_folder = os.path.basename(os.path.dirname(args.resume))
    # decode the defect id back to defect label
    val_n_class, val_insp_label_list, val_ref_image_name_list, val_defect_code, val_defect_decode, \
    val_defect_full_decode, val_val_image_pair_res, val_binary_labels, val_ref_labels, \
    val_insp_labels, val_ref_image_batches, val_insp_image_batches, val_insp_image_name_list, \
    val_ref_image_label_batches, val_insp_image_label_batches, val_binary_label_batchs \
                                                              = get_test_df(args, 
                                                                            train_val_image_pair_path_list, 
                                                                            region, 
                                                                            version_folder, 
                                                                            batch_size, 
                                                                            img_folder,
                                                                            transform,
                                                                            sharpness_save,
                                                                            sharpness_p = 0,
                                                                            calibrate=True
                                                                            )

    
    assert args.resume,  f"必须指定校准模型, args.resume不能为空"
    model = load_resume_model(backbone_arch, output_type, pretrained, n_class, n_units)
    calibrate_models = load_calibrate_model(args, model)
    calibrate_models.cuda()

    criterion_bos = FocalLoss(gamma=gamma, size_average=True, weighted=weighted_loss, label_smooth=smooth)
    criterion_bom = FocalLoss(gamma=gamma, size_average=True, weighted=weighted_loss, label_smooth=smooth)

    if  args.fintune == False:
        print(f"不进行校准fintune")
        p_r_dir = os.path.join(os.path.dirname(args.resume), args.calibrate_dataset, f'T_{args.T_lower}_{args.T_upper}', 'no_fine_tune', 'p_r')
        bos_ECE_results_save_path = os.path.join(os.path.dirname(args.resume), args.calibrate_dataset, f'T_{args.T_lower}_{args.T_upper}', 'no_fine_tune', 'bos_ece')
        bom_ECE_results_save_path = os.path.join(os.path.dirname(args.resume), args.calibrate_dataset,  f'T_{args.T_lower}_{args.T_upper}','no_fine_tune', 'bom_ece')

        os.makedirs(p_r_dir, exist_ok=True)
        calibrate_models.grid_search_set([args.T_lower, args.T_upper], dt=args.dt)
        length_candidate_T = len(calibrate_models)
        defect_decode_bos_dict = {0: 'ok', 1: 'ng'}
        # ECE = ECEer()

        best_bos_acc = 0
        best_bos_acc_thres = 0
        best_bos_acc_T = -1

        max_correct_nums, min_wrong_nums, max_correct_nums_thresholds, min_wrong_nums_thresholds = 0, 1000000, 0.5, 0.5
        max_correct_nums_T,  min_wrong_nums_T = 1.0, 1.0
        calibrate_models.compile_model()

        T_default_infos = {'cur_T_best_thres':{}, 'max_correct_nums':{}, 'min_wrong_nums':{}, 'default_correct_num':{}, 'default_wrong_num':{}}
        
        min_bos_multi_ece = 1000000
        min_bom_multi_ece = 1000000
        min_bos_classwise_ece = 1000000
        min_bom_classwise_ece = 1000000
        min_total_bos_ece = 1000000
        min_total_bom_ece = 1000000



        val_loader = zip(val_ref_image_batches, val_insp_image_batches, val_ref_image_label_batches, val_insp_image_label_batches, val_binary_label_batchs)
        val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions, val_imgs, \
            precision_list, recall_list, f1score_list, thresholds_list, output_bos_th, output_bom_th, bos_labels, bom_labels = valer(val_loader, output_type, calibrate_models, criterion_bos, criterion_bom, gpu_exists, True, use_amp, selection_score)
                
        output_bos_th = output_bos_th.cpu().float()
        output_bom_th = output_bom_th.cpu().float()
        bos_labels = bos_labels.cpu().float()
        bom_labels = bom_labels.cpu().float()

        output_bos_th_T1 = output_bos_th.detach().clone()
        output_bom_th_T1 = output_bom_th.detach().clone()
        ok_w = (bos_labels==0).float().mean()
        ng_w = (bos_labels!=0).float().mean()

        multi_class_weight = {}
        for defect_key, defect_label in defect_decode.items():
            print(defect_key, defect_label)

            cur_defect_w = (bom_labels == defect_key).float().mean()
            multi_class_weight[defect_label] = cur_defect_w

        
        from pathos.multiprocessing import ProcessingPool as Pool
        max_workers = min(len(calibrate_models.candidate_T_arange), os.cpu_count() - 1)
        print(f"start process_T_value")
        s_time = time.time()

        # 选择评估方法（大小写不敏感）
        multi_ECEor, classwise_ECEor, overall_ECEor, classwise_tag, _backend = select_ece_methods(
            args.ECE_calcu_type,
            n_bins=getattr(args, 'n_bins', 15),
            kde_grid_points=getattr(args, 'kde_grid_points', 201),
            kde_bandwidth=getattr(args, 'kde_bandwidth', None),
            kde_reflect=True
        )

        # —— 保持在 GPU，labels 用 long
        device = output_bos_th.device
        output_bos_th = output_bos_th.float().contiguous()
        output_bom_th = output_bom_th.float().contiguous()
        bos_labels = bos_labels.long().contiguous()
        bom_labels = bom_labels.long().contiguous()

        output_bos_th_T1 = output_bos_th.detach().clone()
        output_bom_th_T1 = output_bom_th.detach().clone()

        # 二分类 BOS 的真实占比（与口径无关）
        ok_w = (bos_labels == 0).float().mean().item()
        ng_w = (bos_labels != 0).float().mean().item()

        # 初始化最优记录（用 Python float 更简单）
        min_bos_multi_ece = float('inf'); min_bos_multi_ece_T = None
        min_bom_multi_ece = float('inf'); min_bom_multi_ece_T = None
        min_bos_classwise_ece = float('inf'); min_bos_classwise_ece_T = None
        min_bom_classwise_ece = float('inf'); min_bom_classwise_ece_T = None
        min_total_bos_ece = float('inf'); min_total_bos_ece_T = None
        min_total_bom_ece = float('inf'); min_total_bom_ece_T = None
        min_speical_bos_ng_ece = float('inf'); min_speical_bos_ng_ece_T = None

        for _ in range(length_candidate_T):
            calibrate_models.reflash_T()
            T_value = calibrate_models.T.item()

            # 温度缩放后的 logits
            output_bos_th = output_bos_th_T1 / T_value
            output_bom_th = output_bom_th_T1 / T_value

            # 当前 T 的输出目录
            cur_bos_ECE_results_save_path = os.path.join(
                bos_ECE_results_save_path, args.ECE_calcu_type, f'T_{T_value}')
            cur_bom_ECE_results_save_path = os.path.join(
                bom_ECE_results_save_path, args.ECE_calcu_type, f'T_{T_value}')
            os.makedirs(cur_bos_ECE_results_save_path, exist_ok=True)
            os.makedirs(cur_bom_ECE_results_save_path, exist_ok=True)

            # ---------- (A) Multi 口径 ----------
            bos_multi_val = 0.0; bom_multi_val = 0.0
            if multi_ECEor is not None:
                multi_bos_ece_dict, _ = multi_ECEor(
                    output_bos_th, bos_labels,
                    save_path=cur_bos_ECE_results_save_path,
                    vis_acc_conf=True, vis_bin_ece=True,
                    defect_decode_dict=defect_decode_bos_dict
                )
                multi_bom_ece_dict, _ = multi_ECEor(
                    output_bom_th, bom_labels,
                    save_path=cur_bom_ECE_results_save_path,
                    vis_acc_conf=True, vis_bin_ece=True,
                    defect_decode_dict=defect_decode
                )
                bos_multi_val = float(multi_bos_ece_dict['multiclass_ece'])
                bom_multi_val = float(multi_bom_ece_dict['multiclass_ece'])

                if bos_multi_val < min_bos_multi_ece:
                    min_bos_multi_ece, min_bos_multi_ece_T = bos_multi_val, T_value
                if bom_multi_val < min_bom_multi_ece:
                    min_bom_multi_ece, min_bom_multi_ece_T = bom_multi_val, T_value

            # ---------- (B) Classwise 口径 ----------
            bos_class_val = 0.0; bom_class_val = 0.0
            if classwise_ECEor is not None:
                # BOS（二分类）
                classwise_bos_ece_dict, _ = classwise_ECEor(
                    output_bos_th, bos_labels,
                    save_path=cur_bos_ECE_results_save_path,
                    vis_acc_conf=True, vis_bin_ece=True,
                    defect_decode_dict=defect_decode_bos_dict
                )
                if 'ok' in classwise_bos_ece_dict and 'ng' in classwise_bos_ece_dict:
                    bos_class_val = ok_w * float(classwise_bos_ece_dict['ok']) \
                                + ng_w * float(classwise_bos_ece_dict['ng'])
                    # 记录“ng 类”最小值（你原本的指标）
                    if float(classwise_bos_ece_dict['ng']) < min_speical_bos_ng_ece:
                        min_speical_bos_ng_ece = float(classwise_bos_ece_dict['ng'])
                        min_speical_bos_ng_ece_T = T_value
                    if bos_class_val < min_bos_classwise_ece:
                        min_bos_classwise_ece, min_bos_classwise_ece_T = bos_class_val, T_value

                # BOM（多分类）
                classwise_bom_ece_dict, _ = classwise_ECEor(
                    output_bom_th, bom_labels,
                    save_path=cur_bom_ECE_results_save_path,
                    vis_acc_conf=True, vis_bin_ece=True,
                    defect_decode_dict=defect_decode
                )
                if len(classwise_bom_ece_dict) > 0:
                    # 权重按口径选择
                    if classwise_tag == 'pred':
                        with torch.no_grad():
                            preds_bom = torch.softmax(output_bom_th, dim=1).argmax(dim=1)
                        weight = { defect_decode[k]: float((preds_bom == k).float().mean())
                                for k in defect_decode.keys() }
                    elif classwise_tag == 'true':
                        weight = { defect_decode[k]: float((bom_labels == k).float().mean())
                                for k in defect_decode.keys() }
                    else:
                        weight = None

                    if weight is not None:
                        bom_class_val = 0.0
                        for name, val in classwise_bom_ece_dict.items():
                            if name in weight:
                                bom_class_val += weight[name] * float(val)

                        if bom_class_val < min_bom_classwise_ece:
                            min_bom_classwise_ece, min_bom_classwise_ece_T = bom_class_val, T_value

            # ---------- (C) OVERALL（top-1）可选：只做可视化，不参与你当前总分 ----------
            if overall_ECEor is not None:
                _overall_bos_ece, _overall_bos_mce = overall_ECEor(
                    output_bos_th, bos_labels,
                    save_path=os.path.join(cur_bos_ECE_results_save_path, 'overall'),
                    vis_acc_conf=True
                )
                _overall_bom_ece, _overall_bom_mce = overall_ECEor(
                    output_bom_th, bom_labels,
                    save_path=os.path.join(cur_bom_ECE_results_save_path, 'overall'),
                    vis_acc_conf=True
                )

            # ---------- (D) 组合指标，保持你原先的“classwise + multi”汇总 ----------
            cur_T_total_bos_ece = bos_class_val + bos_multi_val
            cur_T_total_bom_ece = bom_class_val + bom_multi_val

            if cur_T_total_bos_ece < min_total_bos_ece:
                min_total_bos_ece, min_total_bos_ece_T = cur_T_total_bos_ece, T_value
            if cur_T_total_bom_ece < min_total_bom_ece:
                min_total_bom_ece, min_total_bom_ece_T = cur_T_total_bom_ece, T_value

            calibrate_models.reset_index()

        e_time = time.time()
        print(f"spend time: {e_time - s_time:.3f}s")

        # ------- 打包保存结果（转成 Python 标量） -------
        results = {
            'min_bos_multi_ece': min_bos_multi_ece,
            'min_bos_multi_ece_T': min_bos_multi_ece_T,
            'min_bom_multi_ece': min_bom_multi_ece,
            'min_bom_multi_ece_T': min_bom_multi_ece_T,
            'min_bos_classwise_ece': min_bos_classwise_ece,
            'min_bos_classwise_ece_T': min_bos_classwise_ece_T,
            'min_bom_classwise_ece': min_bom_classwise_ece,
            'min_bom_classwise_ece_T': min_bom_classwise_ece_T,
            'min_total_bos_ece': min_total_bos_ece,
            'min_total_bos_ece_T': min_total_bos_ece_T,
            'min_total_bom_ece': min_total_bom_ece,
            'min_total_bom_ece_T': min_total_bom_ece_T,
            f'min_{args.ECE_calcu_type}_bos_ng_ece': min_speical_bos_ng_ece,
            f'min_{args.ECE_calcu_type}_bos_ng_ece_T': min_speical_bos_ng_ece_T
        }
        print(results)

        with open(f'./calibrate_results_{args.ECE_calcu_type}_{args.T_lower}_{args.T_upper}.json',
                'w', encoding='utf-8') as f:
            import json
            json.dump(results, f, ensure_ascii=False, indent=4)

    else:
        print(f"校准fintune模型, 优化校准参数")
        p_r_dir = os.path.join(os.path.dirname(args.resume), args.calibrate_dataset, 'p_r', 'fine_tune')
        os.makedirs(p_r_dir, exist_ok=True)
        kd_config = None
        # define loss function

        if  args.constrative_loss_dist == "Cosin":
            print(f"cl_loss = ContrastiveCosinLoss")
            cl_loss = ContrastiveCosinLoss()
        else:
            print(f"cl_loss = ContrastiveLoss")
            cl_loss = ContrastiveLoss()

        val_loader = zip(val_ref_image_batches, val_insp_image_batches, val_ref_image_label_batches, val_insp_image_label_batches, val_binary_label_batchs)

        (val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass,
        val_label_positions, val_imgs, precision_list, recall_list, f1score_list, thresholds_list,
        output_bos_th, output_bom_th, bos_labels, bom_labels) = \
            valer(val_loader, output_type, calibrate_models, criterion_bos, criterion_bom,
                gpu_exists, True, use_amp, selection_score)
        #   选择校准方式（一次性）
        per_cls = args.ets_per_class or (args.calibrate_method in ('ets_pc','ets-pc'))
        calibrate_models.set_calibrate_method(
            method=args.calibrate_method,      # 'ts' | 'ets' | 'ets_pc'
            per_class=per_cls,
            num_classes_bom=3,   # ETS 必填其一
            num_classes_bos=2      # ETS 必填其一
        )



        calibrate_models.pre_fintune()
        # define optimizer
        optimizer, scheduler, scaler = set_fintune_param(optimizer_type, calibrate_models, init_lr, weight_decay, lr_schedule, use_amp, lr_step_size, lr_gamma)



        print(
            f'Before finetune: val_losses={val_losses}, mclass_val_acc={val_acc_mc}, binary_val_acc={val_acc_bi},  '
            f'binary recall={val_recall_bi:.3f}%, mclass recall={val_recall_mclass:.3f}%')
        
        plot_cur_T_p_r(calibrate_models, p_r_dir, precision_list, recall_list, f1score_list, thresholds_list)

        if args.if_save_test_best_model:
            init_acc_binary, init_acc_mclass = get_test_df_results(calibrate_models, ref_image_batches, insp_image_batches, 
                                                                insp_labels_test_v, binary_labels_test_v,
                                                                    version_name,
                                                                    higher_threshold)
            best_valtest_acc_bi = init_acc_binary
            best_valtest_acc_ml = init_acc_mclass
            s_best_valtest_acc_bi = init_acc_binary
            s_best_valtest_acc_ml = init_acc_mclass
        else:
            best_valtest_acc_bi = 0
            best_valtest_acc_ml = 0
            s_best_valtest_acc_bi = 0
            s_best_valtest_acc_ml = 0
        # start training and save results
        epoch_train_losses = []
        epoch_train_accuracies = []
        epoch_val_losses = []
        epoch_val_accuracies_mclass = []
        epoch_val_accuracies_binary = []
        best_val_accuracies = [0, 0, 0]

        # 记录模型
        dummy_input1 = torch.randn(1, 3, rs_img_size_h, rs_img_size_w).cuda()
        dummy_input2 = torch.randn(1, 3, rs_img_size_h, rs_img_size_w).cuda()

        tb_writer.add_graph(calibrate_models, (dummy_input1, dummy_input2))

        for epoch in tqdm(range(start_epoch, epochs), desc="epoch", initial=start_epoch, total=epochs-start_epoch):
            min_len = min(len(val_ref_image_batches), len(val_insp_image_batches),
                  len(val_ref_image_label_batches), len(val_insp_image_label_batches),
                  len(val_binary_label_batchs))
            val_loader = tqdm(
                zip(val_ref_image_batches, val_insp_image_batches,
                    val_ref_image_label_batches, val_insp_image_label_batches,
                    val_binary_label_batchs),
                total=min_len, desc=f"val at epoch {epoch}", leave=False, dynamic_ncols=True
            )
            val_loss = float('nan')
            val_acc_bi = val_acc_mc = float('nan')
            val_recall_bi = val_recall_mclass = float('nan')
            val_label_positions = {}
            precision_list = recall_list = f1score_list = thresholds_list = []
            output_bos_th = output_bom_th = bos_labels = bom_labels = None
            val_imgs = []  
    
            # 仅在 visualize=True 时会被真正赋值
            # adjust learning rate based on scheduling condition
            # adjust_learning_rate(optimizer, init_lr, epoch, decay_points)
            # train the model for 1 epoch
            # —— 2) 训练一轮 —— 
            if 'CL' in output_type:
                epoch_time_start = time.time()
                train_loss, train_acc = train(train_loader_tqdm, calibrate_models, criterion_bos, criterion_bom,
                                            cl_loss, optimizer, gpu_exists, use_amp, scaler, kd_config)
                epoch_time = time.time() - epoch_time_start
                scheduler.step()

                # —— 3) 验证（CL 分支）——
                try:
                    # 注意：visualize=False 时，val(...) 返回 14 个值
                    (val_loss, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass,
                    val_label_positions,
                    precision_list, recall_list, f1score_list, thresholds_list,
                    output_bos_th, output_bom_th, bos_labels, bom_labels) = val(
                        val_loader, calibrate_models, criterion_bos, criterion_bom, gpu_exists,
                        visualize=False, use_amp=use_amp, selection_score=selection_score
                    )
                except Exception as e:
                    traceback.print_exc()
                    tqdm.write(f"[VAL-CL] 本轮验证异常，已跳过：{e}")

            else:
                epoch_time_start = time.time()
                train_loader_tqdm = tqdm(train_loader, total=len(train_loader),
                                        desc=f"train[{epoch}]", leave=False, dynamic_ncols=True)
                train_loss, train_acc = train_no_cl(train_loader_tqdm, calibrate_models, criterion_bos, criterion_bom,
                                                    cl_loss, optimizer, gpu_exists, use_amp, scaler, kd_config)
                epoch_time = time.time() - epoch_time_start
                scheduler.step()

                # —— 3) 验证（no-CL 分支）——
                try:
                    # 返回 14 个值；左侧用括号包住，支持安全换行
                    (val_loss, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass,
                    val_label_positions,
                    precision_list, recall_list, f1score_list, thresholds_list,
                    _, _, _, _) = val_no_cl(
                        val_loader, calibrate_models, criterion_bos, criterion_bom, gpu_exists,
                        visualize=False, use_amp=use_amp, selection_score=selection_score
                    )
                except Exception as e:
                    traceback.print_exc()
                    tqdm.write(f"[VAL-noCL] 本轮验证异常，已跳过：{e}")

            # —— 4) 统一日志输出（对 NaN 友好）——
            def _fmt(x):
                try:
                    return f"{float(x):.4f}" if x == x else "NaN"
                except Exception:
                    return "NaN"

            tqdm.write(
                f"[ep {epoch}] train_loss={_fmt(train_loss)} | "
                f"val_loss={_fmt(val_loss)} | "
                f"acc_mc={_fmt(val_acc_mc)}% | acc_bi={_fmt(val_acc_bi)}% | "
                f"recall_mc={_fmt(val_recall_mclass)}% | recall_bi={_fmt(val_recall_bi)}%"
            )

            plot_cur_T_p_r(calibrate_models, p_r_dir, precision_list, recall_list, f1score_list, thresholds_list)

            # —— 5) 记录指标（确保 .cpu().numpy() 之前有数值；否则可跳过）——
            epoch_train_losses.append(float(train_loss))
            epoch_train_accuracies.append(train_acc.cpu().numpy())
            epoch_val_losses.append(float(val_loss))
            epoch_val_accuracies_mclass.append(val_acc_mc.cpu().numpy())
            epoch_val_accuracies_binary.append(val_acc_bi.cpu().numpy())

            current_metric_score = val_acc_bi + val_acc_mc + (val_recall_bi + val_recall_mclass) * 150

            tb_writer.add_scalar('train_loss', float(train_loss), epoch)
            tb_writer.add_scalar('train_acc', train_acc.cpu().numpy(), epoch)
            tb_writer.add_scalar('val_loss', float(val_loss) if val_loss == val_loss else 0.0, epoch)
            tb_writer.add_scalar('val_acc_mc', val_acc_mc.cpu().numpy(), epoch)
            tb_writer.add_scalar('val_acc_bi', val_acc_bi.cpu().numpy(), epoch)

            if (save_checkpoint and (run_mode == 'train' or run_mode == 'train_resume')):
                state = {
                    'epoch': epoch + 1,
                    'backbone_model': backbone_arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }

                if current_metric_score > max(best_val_accuracies):
                    best_val_accuracies[0] = current_metric_score
                    checkpoint_path = best_checkpoint_path
                elif current_metric_score > best_val_accuracies[1]:
                    best_val_accuracies[1] = current_metric_score
                    checkpoint_path = best_checkpoint_path.replace('top0', 'top1')
                elif current_metric_score > best_val_accuracies[2]:
                    best_val_accuracies[2] = current_metric_score
                    checkpoint_path = best_checkpoint_path.replace('top0', 'top2')
                else:
                    checkpoint_path = best_checkpoint_path.replace('top0', 'last')


                torch.save(state, checkpoint_path)

                if args.if_save_test_best_model:
                    # 按照目标测试集的acc保留模型
                    acc_binary, acc_mclass = get_test_df_results(calibrate_models, ref_image_batches, insp_image_batches, 
                                                                insp_labels_test_v, binary_labels_test_v,
                                                                version_name, 
                                                                higher_threshold, 
                                                            )
                    
                    if acc_binary > best_valtest_acc_bi and acc_mclass > best_valtest_acc_ml:
                        best_valtest_acc_bi = acc_binary
                        best_valtest_acc_ml = acc_mclass
                        checkpoint_path = best_checkpoint_path.replace('top0', 'bestacc')

                        if acc_binary > s_best_valtest_acc_bi:
                            s_best_valtest_acc_bi = acc_binary
                        
                        if acc_mclass > s_best_valtest_acc_ml:
                            s_best_valtest_acc_ml = acc_mclass

                    elif acc_binary > s_best_valtest_acc_bi:
                        s_best_valtest_acc_bi = acc_binary
                        checkpoint_path = best_checkpoint_path.replace('top0', 'bestbiacc')
                        
                    elif acc_mclass > s_best_valtest_acc_ml:
                        s_best_valtest_acc_ml = acc_mclass
                        checkpoint_path = best_checkpoint_path.replace('top0', 'bestmlacc')
                        
                    torch.save(state, checkpoint_path)

                # save train and val statistics for visualization and checking
                if args.save_train_trace_pdf:
                    res = {'train_loss': epoch_train_losses,
                        'train_acc': epoch_train_accuracies,
                        'val_acc_binary': epoch_val_accuracies_binary,
                        'val_acc_mclass': epoch_val_accuracies_mclass,
                        }

                    # display the train/val curves
                    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

                    axes_flat = axes.flatten()
                    id = 0
                    for key, value in res.items():
                        axes_flat[id].plot(value)
                        axes_flat[id].set_title(key)
                        id += 1

                    plt.tight_layout()
                    plt.savefig(best_checkpoint_path.replace('top0.pth.tar', '.pdf'))
                    plt.close()

            elif (save_checkpoint and (save_checkpoint and run_mode == 'train_ft')):
                state = {
                    'epoch': epoch + 1,
                    'backbone_model': backbone_arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                checkpoint_path = best_checkpoint_path.replace('top0', f'ft{epoch}')
                torch.save(state, checkpoint_path)
            
            if epoch % verbose_frequency == 0:
                print(f'Ep {epoch}: train loss={train_loss:.3f}, train acc={train_acc:.3f}%,'
                    f'val loss={val_loss:.3f}, mclass val acc={val_acc_mc:.3f}%, '
                    f'binary val acc={val_acc_bi:.3f}%,  binary recall={val_recall_bi:.3f}%, mclass recall={val_recall_mclass:.3f}%\n||'                      
                    f'current_metric_score={current_metric_score} '
                    f'best_val_accuracies={best_val_accuracies} '
                    f'best_valtest_acc_bi={best_valtest_acc_bi} '
                    f'best_valtest_acc_ml={best_valtest_acc_ml} '
                    f's_best_valtest_acc_bi={s_best_valtest_acc_bi} '
                    f's_best_valtest_acc_ml={s_best_valtest_acc_ml} '
                    f'time={epoch_time:.3f} s '
                    )

    return val_loader, model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MPB3 for SMT Defect Classification')
    # parser.add_argument('--data', default='/home/robinru/shiyuan_projects/SMTdefectclassification/data/beijingfactory1_smt_defect_data', type=str,
    #                     help='path to image datasets and annotation labels')
    parser.add_argument('--data', default='/home/robinru/shiyuan_projects/smt_defect_data_pipeline/data', type=str,
                        help='path to image datasets and annotation labels')
    parser.add_argument('--ckp', default='./models/checkpoints', type=str,
                        help='path to save and load checkpoint')
    parser.add_argument('--region', default='singlepad', type=str,
                        help='component instance region data to use: singlepad, singlepinpad, padgroup')
    parser.add_argument('--arch', default='cbammobilenetv3small', type=str,
                        help='backbone arch to be used：fcdropoutresnet18, fcdropoutmobilenetv3large, resnet18')
    parser.add_argument('--date', default='231013', type=str,
                        help='data processing date ')
    parser.add_argument('--version_name', default='vtest', type=str,
                        help='model version name')
    parser.add_argument('--resume',
                        default=None,
                        # default='/home/robinru/shiyuan_projects/SMTdefectclassification/models/checkpoints/single_pin_v1.0.0best/single_pinmobilenetv3smallrs12832s42c4val0.15b256_ckp_bestv0.2320.0f16n128j0.4lr0.1dualtop0.pth.tar ',
                        type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--reload_mode',
                        default='full',
                        type=str, help='which mode to load the ckp: full, backbone, skip_mismatch')
    parser.add_argument('--mode', default='train', type=str,
                        help='running mode: validation or training')
    parser.add_argument('--output_type', default='dual2', type=str,
                        help='specify the output tyep: dual, mclass')
    parser.add_argument('--score', default='recall', type=str,
                        help='selection score: recall,  f1')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--k', default=2.5, type=float,
                        help='weighted sampling multiplier')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--resize_w', default=128, type=int,
                        help='resize the input image to this resolution dimension')
    parser.add_argument('--resize_h', default=32, type=int,
                        help='resize the input image to this resolution dimension')
    parser.add_argument('--epochs', default=600, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--worker', default=4, type=int,
                        help='number of workers to run')
    parser.add_argument('--gamma', default=2, type=float,
                        help='gamma value for focal loss')
    parser.add_argument('--smooth', default=0.0, type=float,
                        help='label smoothing ')
    parser.add_argument('-se', '--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-bs', '--batch_size', default=256, type=int,
                        help='mini_batch size')
    parser.add_argument('-nb', '--n_unit_binary', default=128, type=int,
                        help='number of hidden units')
    parser.add_argument('-ji', '--jitter', default=0.4, type=float,
                        help='color jitter')
    parser.add_argument('-nm', '--n_unit_mclass', default=128, type=int,
                        help='number of hidden units')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='initial (base) learning rate: default: 0.025')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum of SGD solver')
    parser.add_argument('-wd', '--weight_decay', default=1e-5, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--optimizer_type', default='sgd', type=str,
                        help='optimizer to use: SGD or RMSProp')
    parser.add_argument('--lr_schedule', default='cosine', type=str,
                        help='learning rate scheduling: cosine or step decay')
    parser.add_argument('--lr_step_size', default=7, type=float,
                        help='step decay size')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='step decay gamma')
    parser.add_argument('-vr', '--val_ratio', default=0.1, type=float,
                        help='ratio of validation ')
    parser.add_argument('-vf', '--verbose_frequency', default=1, type=int,
                        help='frequency to print results')
    parser.add_argument('--amp_lvl', default='f16', type=str,
                        help='precision level used:f32, f16')
    parser.add_argument('--label_conf', default='all', type=str,
                        help='label confidence: all, certain')
    # kd arguments
    parser.add_argument('-kdr', '--kd_ratio', default=0, type=float,
                        help='kd loss weight')
    parser.add_argument('-kdt', '--kd_T', default=2, type=float,
                        help='kd temperature')
    parser.add_argument('--kd_type', default='ce', type=str,
                        help='kd loss type: ce, mse')
    parser.add_argument('--teacher_ckp',
                        default=None,
                        # default='v0.1singlepadbest/singlepadresnetsp18rs6464s42c3val0.1b256_ckp_bestv0.1.2f1certain20.0j0.4lr0.1nb256nm256dual2top2.pth.tar',
                        # default='single_pin_v1.7.0f1/single_pinresnetsp18rs12832s42c3val0.15b256_ckp_bestv1.7.0f1certain20.0j0.4lr0.1nb256nm256dual2top1.pth.tar',
                        type=str,
                        help='teacher ckp path')
    parser.add_argument('-lutp', '--lut_p', default=0, type=float,
                        help='probability of applying lut')
    
    parser.add_argument('-cld', '--constrative_loss_dist', default='Cosin', type=str,
                        help='the calculation function of constrative learning distance')
    parser.add_argument('--lr_schedule_T', default=600, type=int,
                        help='learning rate scheduling: cosine or step decay')
    parser.add_argument('--data_type', default='all', type=str, choices=['2d', '3d', 'all'],
                        help='区分2d/3d关照数据')   
    parser.add_argument('--save_train_trace_pdf', default=False, type=bool,
                        help='是否用pdf保存训练记录')
    
    parser.add_argument('--tb_logdir', default='tensorboard', type=str,
                        help='tensorboard的log路径')

    parser.add_argument('--specila_issue_list', default=[], type=list,
                        help='特供数据集')

    parser.add_argument('--compression_p', default=0, type=float,
                        help='图片压缩概率') 
    parser.add_argument('--if_cp_val', default=0, type=int,
                        help='验证集是否进行压缩', choices=[0, 1])   
    parser.add_argument('--p_range', default=[65, 95], type=float,
                        help='图片压缩程度, 0-100, 值越小压缩程度越大')
    
    # 实验证明PIL的压缩会导致一定程度的失真，所以最好为0，用opencv的压缩;
    parser.add_argument('--select_p', default=0, type=float,
                        help='选择opencv压缩还是PIL压缩, select_p=0为opencv压缩, select_p=1为PIL压缩,0-1之间则随机选择, None则随机选择')
    
    parser.add_argument('--save_traintestval_infos', default=False, type=bool,
                        help='是否保存训练的数据集信息')

    parser.add_argument('--if_save_test_best_model', default=False, type=bool,
                        help='是否根据指定数据集保存模型')
    parser.add_argument('--valdataset_target', default='follow_val', type=str,
                        help='以哪一个测试集为选择模型的标准')
    parser.add_argument('--higher_threshold', default=0.8, type=float,
                        help='以哪一个测试集为选择模型的标准')
    parser.add_argument('--img_color', default='rgb', type=str, choices=['rgb', 'white'])
    parser.add_argument('--light_device', default='all', type=str, choices=['2d', '3d', 'all'], help='2D还是3D光照的结果')
    parser.add_argument('--img_type', default='png', type=str)
    parser.add_argument('--special_data', default='test_csv', type=str, help="选择不同的数据集配置")
    parser.add_argument('--aug_img', default=False, type=bool)
    parser.add_argument('--fintune', default=False, type=bool)
    parser.add_argument('--T_lower', default=0.5, type=float)

    parser.add_argument('--T_upper', default=2, type=float)
    parser.add_argument('--dt', default=0.01, type=float)
    parser.add_argument('--thres_dt', default=0.02, type=float)
    parser.add_argument('--thres_l', default=0.5, type=float)
    parser.add_argument('--thres_u', default=1, type=float)
    parser.add_argument('--default_thres', default=0.8, type=float)

    parser.add_argument('--ets_per_class', default=False,help='ETS 是否为每个类别单独温度')
    parser.add_argument('--calibrate_method',type=str, default='ts', choices=['ts', 'ets', 'ets_pc', 'ets-pc'],help='ts: 温度缩放；ets: 三组件；ets_pc: ETS按类温度')
    parser.add_argument('--calibrate_dataset', default='calibrate_test', type=str)
    parser.add_argument('--ECE_calcu_type', default='Multi_ECE', type=str, choices=['ECE', 'Ada_ECE', 'Multi_ECE', 'Multi_Ada_ECE', 'Classwise_ECE', 'Classwise_Ada_ECE', 'KDE', 'Classwise_KDE', 'Multi_KDE', 'TrueKDE'])

    args = parser.parse_args()
    print(args)
    # run experiment
    val_loader, model = main(args)