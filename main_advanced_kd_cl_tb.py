import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os
from dataloader.image_resampler_pins import mp_weighted_resampler, ImageLoader, stratified_train_val_split, \
    generate_test_pairs, LUT
from dataloader.image_resampler import DiscreteRotate
from torchvision import transforms
from utils.metrics import FocalLoss, ContrastiveLoss, ContrastiveCosinLoss, accuracy, classk_metric
from utils.utilities import adjust_learning_rate, AverageMeter, save_model
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from special_supply_data import special_csvs_name

def get_data_df(aug_train_pair_data_filenames, data_type):
    aug_train_2d_pair_data_list = []
    aug_train_3d_pair_data_list = []

    for train_csv_path in aug_train_pair_data_filenames:
        cur_train_df = pd.read_csv(train_csv_path, index_col=0)
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

def train(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists, use_amp=False, scaler=None, kd_config=None):
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

        if visualize:
            val_imgs.append([img1.cpu().numpy(), img2.cpu().numpy()])
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
    if visualize:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions, val_imgs
    else:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions
    
def val_no_cl(val_loader, model, criterion_bos, criterion_bom, gpu_exists, visualize=False, use_amp=False, selection_score='recall'):

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
                output_bos, output_bom = model(img1, img2)
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

        if visualize:
            val_imgs.append([img1.cpu().numpy(), img2.cpu().numpy()])
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
    if visualize:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions, val_imgs
    else:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions


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
        annotation_filename = os.path.join(annotation_folder, f'trainval_labels_{region}.csv')
        test_annotation_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_{region}_230306v2_model_cleaned.csv')]
        aug_train_pair_data_filenames = [
                                        os.path.join(annotation_folder,
                                                     f'aug_train_pair_labels_{region}_merged.csv'),
                                        os.path.join(annotation_folder,
                                                      f'aug_train_pair_labels_{region}_240329_final_model_cleaned.csv'),
                                         # os.path.join(annotation_folder,
                                         #              f'aug_train_pair_labels_{region}_240403debug_final.csv'),
                                         # os.path.join(annotation_folder,
                                         #              f'aug_train_pair_labels_{region}_240404debug_final.csv'),
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
                                         
                                         ]


        if 'NGonly' in version_name:
            aug_train_pair_data_filenames += [
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_{region}_240424_final_RGB_NG.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_{region}_240428_final_RGB_NG.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_{region}_240429_final_RGB_NG.csv'),
                # os.path.join(annotation_folder,
                #          f'aug_train_pair_labels_singlepinpad_240715_final_RGBng.csv'),
                # os.path.join(annotation_folder,
                #          f'aug_train_pair_labels_singlepinpad_240702_final_RGBng.csv'),
                # os.path.join(annotation_folder,
                #          f'aug_train_pair_labels_singlepinpad_240708_final_RGBng.csv')
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_ng_model_cleaned.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_singlepinpad_240913_final_rgb_ng_model_cleaned.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_singlepinpad_240919_final_rgb_ng_model_cleaned.csv'),
            ]

        elif 'DownSample' in version_name:
            aug_train_pair_data_filenames += [
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_{region}_240424_final_RGBdownsampled.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_{region}_240428_final_RGBdownsampled.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_{region}_240429_final_RGBdownsampled.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_singlepinpad_240715_final_RGB.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_singlepinpad_240702_final_RGB.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_singlepinpad_240708_final_RGB.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_singlepinpad_240725_final_rgb.csv'),
            ]
        else:
            aug_train_pair_data_filenames += [
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_{region}_240424_final_RGB.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_{region}_240428_final_RGB.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_{region}_240429_final_RGB.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_singlepinpad_240715_final_RGB.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_singlepinpad_240702_final_RGB.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_singlepinpad_240708_final_RGB.csv'),
                # os.path.join(annotation_folder,
                #              f'aug_train_pair_labels_singlepinpad_240725_final_rgb.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_singlepinpad_240911v2_final_rgb_model_cleaned.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_singlepinpad_240913_final_rgb_model_cleaned.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_singlepinpad_240919_final_rgb_model_cleaned.csv'),
            ]

        aug_val_pair_data_filenames = [
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
                                                    
                                       ]
        defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
        defect_code_val = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}

    elif region == 'padgroup':
        n_max_pairs_val = 15
        n_max_pairs_train = 20

        annotation_filename = None
        val_annotation_filename = None
        test_annotation_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_{region}_230306v2_rgb_final3.csv')]

        if 'slim' in version_name:
            defect_code = {'ok': 0, 'solder_shortage': 7}
            defect_code_val = {'ok': 0, 'solder_shortage': 7}
        else:
            defect_code = {'ok': 0, 'missing': 1, 'solder_shortage': 7}
            defect_code_val = {'ok': 0, 'missing': 1, 'solder_shortage': 7}
            #
            # defect_code = {'ok': 0, 'missing': 1, 'undersolder': 4, 'pseudosolder': 6, 'solder_shortage': 7}
            # defect_code_val = {'ok': 0, 'missing': 1, 'undersolder': 4, 'pseudosolder': 6, 'solder_shortage': 7}

        aug_train_pair_data_filenames = [
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
                                        
                                        # 金赛点误报数据
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv'),
                                        
                                        # 深圳仁盈误报数据
                                        os.path.join(annotation_folder,
                                                        f'aug_train_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv'),
                                         ]

        if 'NGonly' in version_name:
            aug_train_pair_data_filenames += [
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_240701_final_RGB_ng_model_cleaned_graphite_uncertain.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_240715_final_RGB_ng_model_cleaned_graphite_uncertain.csv')
            ]
        elif '26mhz' in version_name:
            aug_train_pair_data_filenames += [
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_241101_final_rgb_26MHZ_update_241109_model_cleaned_graphite_uncertain.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_241101_final_rgb_mask_26MHZ_update_241109_model_cleaned_graphite_uncertain.csv')
            ]
        else:
            aug_train_pair_data_filenames += [
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_240701_final_RGB_model_cleaned_graphite_uncertain.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_240708_final_RGB_model_cleaned_graphite_uncertain.csv'),
                os.path.join(annotation_folder,
                             f'aug_train_pair_labels_padgroup_240715_final_RGB_model_cleaned_graphite_uncertain.csv'),
            ]

        aug_val_pair_data_filenames = [
                                        os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_{region}_v090nonpaired_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_{region}_v090merged_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_{region}_240407_final_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_240407_final_rgb_final3_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_padgroup_240417_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_padgroup_240418_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_padgroup_240419_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                       
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_240424_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_240428_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_240429_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_train_pair_labels_padgroup_240715_final_RGB_model_cleaned_graphite_uncertain.csv'),

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
                                                        f'aug_test_pair_labels_padgroup_241202_final_rgb_DA512_update_241206_model_cleaned_graphite_uncertain.csv'),
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
                                        os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_250317_final_rgb_DA786_solder_shortage_fp_dropoldpairs.csv'), 
                                        os.path.join(annotation_folder,
                                                        f'aug_test_pair_labels_padgroup_250403_final_rgb_DA852_ry_syh_dropoldpairs.csv'),
                                       ]
        if '26mhz' in version_name:
            aug_train_pair_data_filenames += [os.path.join(annotation_folder,
                                                f'aug_val_pair_labels_padgroup_241101_final_rgb_26MHZ_update_241109_model_cleaned_graphite_uncertain.csv'),

                                              os.path.join(annotation_folder,
                                                           f'aug_val_pair_labels_padgroup_241101_final_rgb_mask_26MHZ_update_241109_model_cleaned_graphite_uncertain.csv')]
        if 'masked' in version_name:
            aug_train_pair_data_filenames += [os.path.join(annotation_folder,
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
                                            os.path.join(annotation_folder,
                                                          f'aug_train_pair_labels_padgroup_250317_final_rgb_mask_DA786_solder_shortage_fp_dropoldpairs.csv'),
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
                                 f'aug_train_pair_labels_padgroup_240701_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                    os.path.join(annotation_folder,
                                 f'aug_train_pair_labels_padgroup_240708_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                    os.path.join(annotation_folder,
                                 f'aug_train_pair_labels_padgroup_240715_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                ]
            aug_val_pair_data_filenames += [os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_padgroup_240419masked_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_padgroup_240419masked_final_RGB_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_padgroup_240424_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_padgroup_240428_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_test_pair_labels_padgroup_240429_final_RGB_mask_model_cleaned_graphite_uncertain.csv'),
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
                                            os.path.join(annotation_folder,
                                                          f'aug_test_pair_labels_padgroup_250317_final_rgb_mask_DA786_solder_shortage_fp_dropoldpairs.csv'), 
                                            ]

        if 'rgbwhite' in version_name:
            aug_train_pair_data_filenames += [
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
                                            os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_250317_final_white_DA786_solder_shortage_fp_dropoldpairs.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_train_pair_labels_padgroup_250403_final_white_DA852_ry_syh_dropoldpairs.csv'),
                                             ]
            aug_val_pair_data_filenames += [
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
                                            os.path.join(annotation_folder,
                                                           f'aug_test_pair_labels_padgroup_250317_final_white_DA786_solder_shortage_fp_dropoldpairs.csv'),
                                            os.path.join(annotation_folder,
                                                           f'aug_test_pair_labels_padgroup_250403_final_white_DA852_ry_syh_dropoldpairs.csv'),
                                                           
                                           ]
            if '26mhz' in version_name:
                aug_train_pair_data_filenames +=[
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_241101_final_white_26MHZ_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_train_pair_labels_padgroup_241101_final_white_mask_26MHZ_update_241109_model_cleaned_graphite_uncertain.csv'),
 
                ]

                aug_val_pair_data_filenames += [
                                            os.path.join(annotation_folder,
                                                         f'aug_val_pair_labels_padgroup_241101_final_white_26MHZ_update_241109_model_cleaned_graphite_uncertain.csv'),
                                            os.path.join(annotation_folder,
                                                         f'aug_val_pair_labels_padgroup_241101_final_white_mask_26MHZ_update_241109_model_cleaned_graphite_uncertain.csv'),
                ]

            if 'masked' in version_name:
                aug_train_pair_data_filenames += [os.path.join(annotation_folder,
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
                                                                                                            
                                                 os.path.join(annotation_folder,
                                                              f'aug_train_pair_labels_padgroup_250317_final_white_mask_DA786_solder_shortage_fp_dropoldpairs.csv'),                                                              
                                                  ]
                aug_val_pair_data_filenames += [os.path.join(annotation_folder,
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
                                                os.path.join(annotation_folder,
                                                              f'aug_test_pair_labels_padgroup_250317_final_white_mask_DA786_solder_shortage_fp_dropoldpairs.csv'),                                                                                                                     
                                                ]


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
        annotation_filename = None
        val_annotation_filename = None
        test_annotation_filenames = [os.path.join(annotation_folder, f'bbtest_labels_pairs_{region}_230306v2_model_cleaned.csv')]

        aug_train_pair_data_filenames = [
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
                    
                                         ]
        aug_val_pair_data_filenames = [
                                        os.path.join(annotation_folder,f'aug_val_pair_labels_{region}_merged_model_cleaned.csv'),
                                        os.path.join(annotation_folder, f'aug_val_pair_labels_{region}_240329_final_model_cleaned.csv'),
                                        os.path.join(annotation_folder, f'aug_test_pair_labels_{region}_240329_final_model_cleaned.csv'),

                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_240428_final_RGB_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_240429_final_RGB_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_val_pair_labels_{region}_240808_final_RGB_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                    f'aug_test_pair_labels_{region}_241018_final_rgb_model_cleaned.csv'),
                                       # 3D lighting  
                                       os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_singlepad_241103_final_rgb_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_singlepad_241111_final_rgb_D433_model_cleaned.csv'),
                                    #    os.path.join(annotation_folder,
                                    #                   f'aug_test_pair_labels_singlepad_241114_final_rgb_DA465.csv'),
                                       os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_singlepad_241114_final_rgb_DA472_model_cleaned.csv'),

                                       os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_singlepad_241126_final_rgb_DA505_model_cleaned.csv'),
                                       os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_singlepad_241126_final_rgb_DA507_model_cleaned.csv'),

                                       os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_singlepad_241206_final_rgb_DA534.csv'),
                                       os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_singlepad_241206_final_rgb_DA519.csv'),             
                                       os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_singlepad_250109_final_rgb_DA627-629_dropoldpairs.csv'),
                                       os.path.join(annotation_folder,
                                                      f'aug_test_pair_labels_singlepad_250110_final_rgb_DA629-2_dropoldpairs.csv'),
                                       ]


    if len(args.specila_issue_list) > 0:
        if ',' in args.specila_issue_list or ' ' in args.specila_issue_list or \
        (len(args.specila_issue_list[-1]) == 1 and len(args.specila_issue_list[0]) == 1):
            specila_issues = ''.join(args.specila_issue_list).split(',')
        else:
            specila_issues = args.specila_issue_list

        for jira_issue_name in specila_issues:
            jira_issue_name = jira_issue_name.strip()
            print(f"special supply data: {jira_issue_name}")
            special_train_csvs = special_csvs_name[region][jira_issue_name]['train']
            special_val_csvs = special_csvs_name[region][jira_issue_name]['val']

            for special_train_csv in special_train_csvs:
                aug_train_pair_data_filenames.append(os.path.join(annotation_folder, special_train_csv))

            for special_val_csv in special_val_csvs:
                aug_val_pair_data_filenames.append(os.path.join(annotation_folder, special_val_csv))

    n_class = len(defect_code)
    defect_code_link = {v: i for i, v in enumerate(defect_code.values())}
    defect_decode = {v: k for k, v in defect_code_link.items()}
    defect_class_considered = list(defect_code.values())
    defect_class_considered_val = list(defect_code_val.values())
    print(f'train data csvs ={aug_train_pair_data_filenames}')
    print(f'val data csvs ={aug_val_pair_data_filenames}')

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
    # load data and perform stratified sampling to form val and train dataset
    if annotation_filename is not None:
        print(f'processing annotation_filename')
        train_annotation_df, val_annotation_df = stratified_train_val_split(annotation_filename, val_ratio=val_ratio,
                                                            defect_code=defect_code_link, region=region,
                                                            groupby=f'defect_label', seed=seed,
                                                            label_confidence=label_confidence,
                                                            verbose=True)
        Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = mp_weighted_resampler(
            train_annotation_df, n_max_pairs=n_max_pairs_train, batch_size=batch_size, k=k_factor)

        # # TODO: tmpt saving paired data
        # train_pair_data = [x + ytrain_resampled[i] + [ybinary_resampled[i], material_train_resampled[i]]
        #                  for i, x in enumerate(Xtrain_resampled)]
        # train_pair_data_df = pd.DataFrame(train_pair_data,
        #                             columns=['ref_image', 'insp_image', 'ref_y', 'insp_y',  'binary_y', 'material_id'])
        #
        # train_pair_data_df['ref_y_raw'] = [defect_decode[y] for y in train_pair_data_df['ref_y']]
        # train_pair_data_df['insp_y_raw'] = [defect_decode[y] for y in train_pair_data_df['insp_y']]
        # print(len(train_pair_data_df))
        # train_pair_data_df.to_csv(os.path.join(annotation_folder_mnt, f'aug_train_pair_labels_{region}_nonpaired_rgb.csv'))
        if args.data_type == '3d':
            Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = [], [], [], []
        # annotation_filename = os.path.join(annotation_folder, f'trainval_labels_{region}.csv') 这个全是2D数据

    else:
        print(f'no annotation_filename')
        val_annotation_df = None
        Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = [], [], [], []

    if run_mode.startswith('train'):
        # perform weighted resampling on training data
        if run_mode == 'train_ft':
            Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = [], [], [], []

        if aug_train_pair_data_filenames is not None:
            aug_train_pair_data_df_raw = get_data_df(aug_train_pair_data_filenames, args.data_type)

            # aug_train_pair_data_df_raw = pd.concat([pd.read_csv(f, index_col=0) for f in aug_train_pair_data_filenames]).reset_index(drop=True)

            if label_confidence == 'certain':
                aug_train_pair_data_df_certain = aug_train_pair_data_df_raw[
                    (aug_train_pair_data_df_raw['confidence'] == 'certain') | (
                                aug_train_pair_data_df_raw['confidence'] == 'unchecked')].copy()
                print(f'aiug train pair certain = {len(aug_train_pair_data_df_certain)} out of {len(aug_train_pair_data_df_raw)}')
                aug_train_pair_data_df_raw = aug_train_pair_data_df_certain
            


            aug_train_pair_data_df = aug_train_pair_data_df_raw[[y in defect_class_considered for
                                                             y in aug_train_pair_data_df_raw['insp_y_raw']]].copy().reset_index(drop=True)
            n_aug = len(aug_train_pair_data_df)
            print(f'aug train with {n_aug} imag_pairs')
            for i, datum in aug_train_pair_data_df.iterrows():
                Xtrain_resampled.append([datum['ref_image'], datum['insp_image']])
                insp_y_raw = datum['insp_y_raw']
                insp_y = defect_code_link[insp_y_raw]
                ytrain_resampled.append([datum['ref_y'], insp_y])
                ybinary_resampled.append(datum['binary_y'])
                material_train_resampled.append(datum['material_id'])

        train_pair_data = [x + ytrain_resampled[i] + ytrain_resampled[i] + [ybinary_resampled[i], material_train_resampled[i]]
                         for i, x in enumerate(Xtrain_resampled)]
        train_pair_data_df = pd.DataFrame(train_pair_data,
                                    columns=['ref_image', 'insp_image', 'ref_y', 'insp_y', 'ref_y_raw', 'insp_y_raw',
                                             'binary_y', 'material_id'])
        train_pair_data_df.to_csv(os.path.join(annotation_folder_mnt, f'annotation_train_pair_labels_{region}.csv'))
        print('train data histograms')
        print(train_pair_data_df.groupby(['insp_y']).count())
        print(f'n_train_pairs = {len(material_train_resampled)}')
        train_dataset = ImageLoader(img_folder, Xtrain_resampled, ytrain_resampled, ybinary_resampled,
                                    material_train_resampled,
                                    transform=transform_separate,
                                    transform_same=transform_same,
                                    transform_sync=transform_sync)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=n_workers, pin_memory=True)
        # n_train = len(train_annotation_df)

    # refix random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # # generate pair inputs for test data
    test_pair_data_df_raw = get_data_df(test_annotation_filenames, args.data_type)

    # test_pair_data_df_raw = pd.concat(
    #     [pd.read_csv(f, index_col=0) for f in test_annotation_filenames]).reset_index(drop=True)
    if label_confidence == 'certain':
        test_pair_data_df_certain = test_pair_data_df_raw[
            (test_pair_data_df_raw['confidence'] == 'certain') | (
                    test_pair_data_df_raw['confidence'] == 'unchecked')].copy()
        print(f'test pair certain = {len(test_pair_data_df_certain)} out of {len(test_pair_data_df_raw)}')
        test_pair_data_df_raw = test_pair_data_df_certain
    test_pair_data_df = test_pair_data_df_raw[[y in defect_class_considered_val for
                                                     y in test_pair_data_df_raw['insp_y_raw']]].copy().reset_index(drop=True)
    test_pair_data_df['insp_y'] = [defect_code_link[yraw] for yraw in test_pair_data_df['insp_y_raw']]
    print('test data histograms')
    print(test_pair_data_df.groupby(['insp_y']).count())

    Xtest, ytest, ytestraw, ybinary_test, material_test = [], [], [], [], []
    for i, datum in test_pair_data_df.iterrows():
        Xtest.append([datum['ref_image'], datum['insp_image']])
        insp_y_raw = datum['insp_y_raw']
        insp_y = defect_code_link[insp_y_raw]
        ytest.append([datum['ref_y'], insp_y])
        ytestraw.append([datum['ref_y_raw'], datum['insp_y_raw']])
        ybinary_test.append(datum['binary_y'])
        material_test.append(datum['material_id'])
    print(f'test: n_defect_pairs={np.sum(ybinary_test)}, n_ok_pairs={len(ybinary_test) - np.sum(ybinary_test)}, n_total_pairs={len(ybinary_test)}')
    # generate pair inputs for test data
    test_dataset = ImageLoader(img_folder, Xtest, ytest, ybinary_test, material_test,
                               transform=transforms.Compose([
                                   LUT(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'),
                                       lut_p=lut_p),
                                   transforms.Resize((rs_img_size_h, rs_img_size_w)),
                                   transforms.Normalize(mean=transform_mean,
                                                        std=transform_std),
                               ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=(batch_size * 2),
                                              shuffle=False, num_workers=n_workers, pin_memory=True)

    # generate pair inputs for validation data
    Xval, yval, yvalraw, ybinary_val, material_val = [], [], [], [], []
    if val_ratio == 0:
        if val_annotation_filename is not None:

            # val_annotation_df_raw = pd.read_csv(val_annotation_filename, index_col=0)
            val_annotation_df_raw = get_data_df([val_annotation_filename], args.data_type)

            if label_confidence == 'certain':
                val_annotation_df_certain = val_annotation_df_raw[
                    (val_annotation_df_raw['confidence'] == 'certain') | (
                            val_annotation_df_raw['confidence'] == 'unchecked')].copy()
                print(f'aug val pair certain = {len(val_annotation_df_certain)} out of {len(val_annotation_df_raw)}')
                val_annotation_df_raw = val_annotation_df_certain

            val_annotation_df = val_annotation_df_raw[[y in defect_class_considered_val for
                                                             y in
                                                             val_annotation_df_raw['original_y']]].copy().reset_index(
                drop=True)
            val_annotation_df['original_y'] = val_annotation_df['y']
            val_annotation_df['y'] = [defect_code_link[y] for y in val_annotation_df['original_y']]
            val_annotation_df['X_file_path'] = val_annotation_df[f'image_path']

            Xval, yval, yvalraw, ybinary_val, material_val = generate_test_pairs(val_annotation_df,
                                                                                 n_max_pairs=n_max_pairs_val,
                                                                                 reference_annotation_df=None)
            # TODO: tmpt saving paired data
            # val_pair_data = [x + yval[i] + [ybinary_val[i], material_val[i]] for i, x in enumerate(Xval)]
            # val_pair_data_df = pd.DataFrame(val_pair_data,
            #                                 columns=['ref_image', 'insp_image', 'ref_y', 'insp_y',
            #                                          'binary_y', 'material_id'])
            # val_pair_data_df['ref_y_raw'] = [defect_decode[y] for y in val_pair_data_df['ref_y']]
            # val_pair_data_df['insp_y_raw'] = [defect_decode[y] for y in val_pair_data_df['insp_y']]
            # print(len(val_pair_data_df))
            # val_pair_data_df.to_csv(os.path.join(annotation_folder_mnt, f'aug_val_pair_labels_{region}_nonpaired_rgb.csv'))
    else:
        if region == 'padgroup':
            # val_annotation_df来自上面annotation_filename划分出来的
            if val_annotation_filename is not None:
                val_annotation_df_filtered = val_annotation_df[[y in defect_class_considered_val for
                                                                y in
                                                                val_annotation_df['original_y']]].copy().reset_index(
                    drop=True)
                # val_annotation_df_raw = pd.read_csv(val_annotation_filename, index_col=0)
                val_annotation_df_raw = get_data_df([val_annotation_filename], args.data_type)

                if label_confidence == 'certain':
                    val_annotation_df_certain = val_annotation_df_raw[
                        (val_annotation_df_raw['confidence'] == 'certain') | (
                                val_annotation_df_raw['confidence'] == 'unchecked')].copy()
                    print(f'aug val pair certain = {len(val_annotation_df_certain)} out of {len(val_annotation_df_raw)}')
                    val_annotation_df_raw = val_annotation_df_certain
                val_annotation_df_raw['original_y'] = val_annotation_df_raw['y']

                val_annotation_df_new = val_annotation_df_raw[[y in defect_class_considered_val for y in
                                                                 val_annotation_df_raw['original_y']]].copy().reset_index(
                    drop=True)
                val_annotation_df_new['y'] = [defect_code_link[y] for y in val_annotation_df_new['original_y']]
                val_annotation_df_new['X_file_path'] = val_annotation_df_new[f'image_path']
                val_annotation_df = pd.concat([val_annotation_df_new[val_annotation_df_filtered.columns], val_annotation_df_filtered]).reset_index(drop=True)

                Xval, yval, yvalraw, ybinary_val, material_val = generate_test_pairs(val_annotation_df,
                                                                                 n_max_pairs=n_max_pairs_val,
                                                                                 reference_annotation_df=train_annotation_df)

        else:
            if val_annotation_df is not None:
                val_annotation_df_filtered = val_annotation_df[[y in defect_class_considered_val for
                                                                y in
                                                                val_annotation_df['original_y']]].copy().reset_index(
                    drop=True)
                Xval, yval, yvalraw, ybinary_val, material_val = generate_test_pairs(val_annotation_df_filtered,
                                                                                    n_max_pairs=n_max_pairs_val,
                                                                                 reference_annotation_df=train_annotation_df)
            else:
                print(f'no val_annotation_df')
                Xval, yval, yvalraw, ybinary_val, material_val = [], [], [], [], []
            # TODO: tmpt saving paired data
            # val_pair_data = [x + yval[i] + [ybinary_val[i], material_val[i]] for i, x in enumerate(Xval)]
            # val_pair_data_df = pd.DataFrame(val_pair_data,
            #                                 columns=['ref_image', 'insp_image', 'ref_y', 'insp_y',
            #                                          'binary_y', 'material_id'])
            # train_pair_data_df['ref_y_raw'] = [defect_decode[y] for y in val_pair_data_df['ref_y']]
            # train_pair_data_df['insp_y_raw'] = [defect_decode[y] for y in val_pair_data_df['insp_y']]
            # print(len(val_pair_data_df))
            # val_pair_data_df.to_csv(os.path.join(annotation_folder_mnt, f'aug_val_pair_labels_{region}_nonpairedsplit_rgb.csv'))

    # # generate pair inputs for test data
    if aug_val_pair_data_filenames is not None:
        # aug_val_pair_data_df_raw = pd.concat(
        #     [pd.read_csv(f, index_col=0) for f in aug_val_pair_data_filenames]).reset_index(drop=True)
        aug_val_pair_data_df_raw = get_data_df(aug_val_pair_data_filenames, args.data_type)
        

        if label_confidence == 'certain':
            aug_val_pair_data_df_certain = aug_val_pair_data_df_raw[
                (aug_val_pair_data_df_raw['confidence'] == 'certain') | (
                        aug_val_pair_data_df_raw['confidence'] == 'unchecked')].copy()
            print(f'aug val pair certain = {len(aug_val_pair_data_df_certain)} out of {len(aug_val_pair_data_df_raw)}')
            aug_val_pair_data_df_raw = aug_val_pair_data_df_certain
        aug_val_pair_data_df = aug_val_pair_data_df_raw[[y in defect_class_considered_val for
                                                         y in
                                                         aug_val_pair_data_df_raw['insp_y_raw']]].copy().reset_index(drop=True)
        n_aug_val = len(aug_val_pair_data_df)
        print(f'aug val withh {n_aug_val} image pairs')
        for i, datum in aug_val_pair_data_df.iterrows():
            Xval.append([datum['ref_image'], datum['insp_image']])
            insp_y_raw = datum['insp_y_raw']
            insp_y = defect_code_link[insp_y_raw]
            yval.append([datum['ref_y'], insp_y])
            ybinary_val.append(datum['binary_y'])
            material_val.append(datum['material_id'])

    val_pair_data = [x + yval[i] + [ybinary_val[i], material_val[i]] for i, x in enumerate(Xval)]
    val_pair_data_df = pd.DataFrame(val_pair_data,
                                    columns=['ref_image', 'insp_image', 'ref_y', 'insp_y',
                                             'binary_y', 'material_id'])
    # val_pair_data_df.to_csv(os.path.join(annotation_folder_mnt, f'annotation_val_pair_labels_{region}.csv'))
    print('val data histograms')

    print(val_pair_data_df.groupby(['insp_y']).count())
    # aug val_data with test ng pairs
    for i, datum in test_pair_data_df.iterrows():
        if datum['binary_y'] == False:
            continue
        Xval.append([datum['ref_image'], datum['insp_image']])
        insp_y_raw = datum['insp_y_raw']
        insp_y = defect_code_link[insp_y_raw]
        yval.append([datum['ref_y'], insp_y])
        ybinary_val.append(datum['binary_y'])
        material_val.append(datum['material_id'])

        # assert False
    val_dataset = ImageLoader(img_folder, Xval, yval, ybinary_val, material_val,
                              transform=transforms.Compose([
                                  LUT(lut_path=os.path.join(img_folder, 'merged_annotation', 'insp.lut.png'),
                                      lut_p=lut_p),
                                  transforms.Resize((rs_img_size_h, rs_img_size_w)),
                                  transforms.Normalize(mean=transform_mean,
                                                       std=transform_std)
                              ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=(batch_size * 2),
                                             shuffle=False, num_workers=n_workers, pin_memory=True)

    print(f'n_train={len(ybinary_resampled)}, n_val={len(ybinary_val)}, n_test={len(ybinary_test)}, n_total={len(ybinary_val)+len(ybinary_test)+len(ybinary_resampled)}')

    # define loss function
    criterion_bos = FocalLoss(gamma=gamma, size_average=True, weighted=weighted_loss, label_smooth=smooth)
    criterion_bom = FocalLoss(gamma=gamma, size_average=True, weighted=weighted_loss, label_smooth=smooth)

    if  args.constrative_loss_dist == "Cosin":
        print(f"cl_loss = ContrastiveCosinLoss")
        cl_loss = ContrastiveCosinLoss()
    else:
        print(f"cl_loss = ContrastiveLoss")
        cl_loss = ContrastiveLoss()

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
    # define teacher model if use KD
    if kd_ratio > 0:
        teacher_model_path = os.path.join(ckp_path, args.teacher_ckp)

        teacher_model_ckp_name = teacher_model_path.split('/')[-1]
        teacher_backbone_arch = teacher_model_ckp_name.split('rs')[0].split(region)[-1]
        if 'top' in version_name:
            teacher_n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                       teacher_model_ckp_name.split('f16')[-1].split('top')[0].split('nb')[-1].split('nm')]
        else:
            teacher_n_units = [int(n) if 'dual' not in n else int(n.split('dual')[0]) for n in
                       teacher_model_ckp_name.split('f16')[-1].split('last')[0].split('nb')[-1].split('nm')]

        if 'resnetsp' in backbone_arch and 'CL' not in output_type:
            from models.MPB3 import MPB3net
            teacher_model = MPB3net(backbone=teacher_backbone_arch, pretrained=False, n_class=n_class, n_units=teacher_n_units, output_form=output_type)
        elif 'CL' in output_type:
            from models.MPB3_ConstLearning import MPB3net
            teacher_model = MPB3net(backbone=teacher_backbone_arch, pretrained=False, n_class=n_class, n_units=teacher_n_units, output_form=output_type)

        else:
            from models.MPB3 import MPB3net
            teacher_model = MPB3net(backbone=teacher_backbone_arch, pretrained=False, n_class=n_class, n_units=teacher_n_units, output_form=output_type)

        print(f'=> Loading teacher checkpoint {teacher_model_path}')
        teacher_checkpoint = torch.load(teacher_model_path)
        teacher_model.load_state_dict(teacher_checkpoint['state_dict'])
        teacher_model.cuda()

        val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions, val_imgs = val(
            val_loader, teacher_model, criterion_bos, criterion_bom, gpu_exists, visualize=True, use_amp=use_amp,
            selection_score=selection_score)

        test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = val(
            test_loader,
            teacher_model,
            criterion_bos,
            criterion_bom,
            gpu_exists, use_amp=use_amp, selection_score=selection_score)

        print(
            f'Teacher model: val_losses={val_losses}, mclass_val_acc={val_acc_mc}, binary_val_acc={val_acc_bi},  '
            f'binary recall={val_recall_bi:.3f}%, mclass recall={val_recall_mclass:.3f}%, '
            f'binary test acc={test_acc_bi:.3f}%, mclass test acc={test_acc_mc:.3f}%,'
            f'binary test recall={test_recall_bi:.3f}%, mclass test recall={test_recall_mclass:.3f}%,')

        kd_config = {}
        kd_config['kd_ratio'] = kd_ratio
        kd_config['teacher_model'] = teacher_model
        kd_config['kd_type'] = kd_type
        kd_config['temperature'] = kd_T
    else:
        kd_config = None

    # define optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-4)
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=weight_decay)

    elif optimizer_type == "sgdft":
        output_layers = [param for param in model.head_bos.parameters()] + [param for param in
                                                                            model.head_bom.parameters()]
        optimizer = torch.optim.SGD([{'params': model.cnn_encoder.parameters(), 'lr': init_lr / 10},
                                     {'params': output_layers}],
                                    lr=init_lr, momentum=0.9, weight_decay=weight_decay)

    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=init_lr, momentum=0.9,
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

    if run_mode.startswith('train'):
        if gpu_exists:
            model.cuda()

        if args.resume is None:
            print('=> Traing model from scratch')

        else:
            print(f' Resume training with mode={args.reload_mode}=> Loading checkpoint {args.resume}')
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
            if run_mode == 'train_resume':
                assert args.reload_mode == 'full'
                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])

            model.cuda()
            print(f'=> Loaded checkpoint {args.resume} at epoch {start_epoch}')

            if "CL" in output_type:
                val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions, val_imgs = val(
                    val_loader, model, criterion_bos, criterion_bom, gpu_exists, visualize=True, use_amp=use_amp, selection_score=selection_score)

                test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = val(
                    test_loader,
                    model,
                    criterion_bos,
                    criterion_bom,
                    gpu_exists, use_amp=use_amp, selection_score=selection_score)
            else:
                val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions, val_imgs = val_no_cl(
                    val_loader, model, criterion_bos, criterion_bom, gpu_exists, visualize=True, use_amp=use_amp, selection_score=selection_score)

                test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = val_no_cl(
                    test_loader,
                    model,
                    criterion_bos,
                    criterion_bom,
                    gpu_exists, use_amp=use_amp, selection_score=selection_score)

            print(
                f'Before finetune: val_losses={val_losses}, mclass_val_acc={val_acc_mc}, binary_val_acc={val_acc_bi},  '
                f'binary recall={val_recall_bi:.3f}%, mclass recall={val_recall_mclass:.3f}%, '
                f'binary test acc={test_acc_bi:.3f}%, mclass test acc={test_acc_mc:.3f}%,'
                f'binary test recall={test_recall_bi:.3f}%, mclass test recall={test_recall_mclass:.3f}%,')

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

        tb_writer.add_graph(model, (dummy_input1, dummy_input2))

        for epoch in range(start_epoch, epochs):
            # adjust learning rate based on scheduling condition
            # adjust_learning_rate(optimizer, init_lr, epoch, decay_points)
            # train the model for 1 epoch
            if 'CL' in output_type:
                epoch_time_start = time.time()
                train_loss, train_acc = train(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists,
                                            use_amp, scaler, kd_config)
                epoch_time = time.time() - epoch_time_start
                scheduler.step()
                # validate the model
                # with torch.no_grad():
                val_loss, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions = val(val_loader,
                                                                                                            model,
                                                                                                            criterion_bos,
                                                                                                            criterion_bom,
                                                                                                            gpu_exists,
                                                                                                            use_amp=use_amp,
                                                                                                            selection_score=selection_score)

                test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = val(
                    test_loader, model, criterion_bos, criterion_bom, gpu_exists, use_amp=use_amp, selection_score=selection_score)

            else:
                epoch_time_start = time.time()
                train_loss, train_acc = train_no_cl(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists,
                                            use_amp, scaler, kd_config)
                epoch_time = time.time() - epoch_time_start
                scheduler.step()
                # validate the model
                # with torch.no_grad():
                val_loss, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions = val_no_cl(val_loader,
                                                                                                            model,
                                                                                                            criterion_bos,
                                                                                                            criterion_bom,
                                                                                                            gpu_exists,
                                                                                                            use_amp=use_amp,
                                                                                                            selection_score=selection_score)

                test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = val_no_cl(
                    test_loader, model, criterion_bos, criterion_bom, gpu_exists, use_amp=use_amp, selection_score=selection_score)

            # record loss and other metrics
            epoch_train_losses.append(train_loss)
            epoch_train_accuracies.append(train_acc.cpu().numpy())
            epoch_val_losses.append(val_loss)
            epoch_val_accuracies_mclass.append(val_acc_mc.cpu().numpy())
            epoch_val_accuracies_binary.append(val_acc_bi.cpu().numpy())

            if epoch % verbose_frequency == 0:
                print(f'Ep {epoch}: train loss={train_loss:.3f}, train acc={train_acc:.3f}%,'
                      f'val loss={val_loss:.3f}, mclass val acc={val_acc_mc:.3f}%, '
                      f'binary val acc={val_acc_bi:.3f}%,  binary recall={val_recall_bi:.3f}%, mclass recall={val_recall_mclass:.3f}%||'
                      f'binary test acc={test_acc_bi:.3f}%, mclass test acc={test_acc_mc:.3f}%, '
                      f'binary test recall={test_recall_bi:.3f}%, mclass test recall={test_recall_mclass:.3f}%, '
                      f'time={epoch_time:.3f} s')

            current_metric_score = val_acc_bi + val_acc_mc + (val_recall_bi + val_recall_mclass) * 150
            
            tb_writer.add_scalar('train_loss', train_loss, epoch)
            tb_writer.add_scalar('train_acc', train_acc.cpu().numpy(), epoch)
            tb_writer.add_scalar('val_loss', val_loss, epoch)
            tb_writer.add_scalar('val_acc_mc', val_acc_mc.cpu().numpy(), epoch)
            tb_writer.add_scalar('val_acc_bi', val_acc_bi.cpu().numpy(), epoch)
            tb_writer.add_scalar('current_metric_score', current_metric_score, epoch)

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
                # else:
                #     checkpoint_path = best_checkpoint_path.replace('top0', 'top3')

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

        return val_loader, model

    else:
        print('Not implemented')


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
    
    args = parser.parse_args()
    print(args)
    # run experiment
    val_loader, model = main(args)