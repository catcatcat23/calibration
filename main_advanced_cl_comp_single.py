import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os
from dataloader.image_resampler import mp_weighted_resampler, ImageLoader3, stratified_train_val_split, \
    generate_val_pairs, DiscreteRotate, generate_test_pairs
from torchvision import transforms
from utils.metrics import FocalLoss, ContrastiveLoss, ContrastiveCosinLoss,accuracy, classk_metric
from utils.utilities import adjust_learning_rate, AverageMeter, save_model
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def train(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists, use_amp=False, scaler=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    for batch_id, (img1, img2, label1, label2, binary_y, position) in enumerate(train_loader):

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
                loss_binary = criterion_bos(output_bos, label_binary) * output_bom.shape[1]
                loss = (loss_binary + loss_mclass) / 2
                if (batch_id > 0) and (batch_id % (len(train_loader) - 1) == 0):
                    print(f'binary_loss={loss_binary}, mclass_loss={loss_mclass}， n_class={output_bom.shape}, loss_contrastive={loss_contrastive}')
            else:
                loss = loss_mclass

            loss = 4 * loss + loss_contrastive # loss微调时一般时0.5， loss_contrastive大约为1，并且loss时最终的结果，对backbone的影响可能xi所以要给loss一定的权重

        # compute gradient and perform gradient update
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # update record
        acc1 = accuracy(output_bom, label2, topk=(1,))
        batch_size = img1.size(0)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc1[0], batch_size)

    return losses.avg, accuracies.avg

def train_no_cl(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists, use_amp=False, scaler=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    for batch_id, (img1, img2, label1, label2, binary_y, position) in enumerate(train_loader):

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
                loss_binary = criterion_bos(output_bos, label_binary) * output_bom.shape[1]
                loss = (loss_binary + loss_mclass) / 2
                if (batch_id > 0) and (batch_id % (len(train_loader) - 1) == 0):
                    print(f'binary_loss={loss_binary}, mclass_loss={loss_mclass}， n_class={output_bom.shape}')
            else:
                loss = loss_mclass

        # compute gradient and perform gradient update
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # update record
        acc1 = accuracy(output_bom, label2, topk=(1,))
        batch_size = img1.size(0)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc1[0], batch_size)

    return losses.avg, accuracies.avg

def train_cl_pre(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists, use_amp=False, scaler=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    for batch_id, (img1, img2, label1, label2, binary_y, position) in enumerate(train_loader):

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
            loss = cl_loss(output_bos, output_bom, label_binary)

        # compute gradient and perform gradient update
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # update record
        # acc1 = accuracy(output_bom, label2, topk=(1,))
        batch_size = img1.size(0)
        losses.update(loss.item(), batch_size)
        # accuracies.update(acc1[0], batch_size)

    return losses.avg  #, accuracies.avg

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

    for batch_id, (img1, img2, label1, label2, binary_y, position) in enumerate(val_loader):

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
            val_label_positions.append([output_bos_np, output_bom.detach().cpu().numpy(),
                                        label1, label2, label_binary, position])
        else:
            val_label_positions.append([label1, label2, label_binary, position])

    if output_bos is not None:
        label_binary_all = torch.cat(label_binary_list, dim=0)
        output_bos_all = torch.cat(output_bos_list, dim=0)
        recall_bi = classk_metric(output_bos_all, label_binary_all, 1, input_type='torch')[metric_idx]
    else:
        recall_bi = 0

    output_bom_all = torch.cat(output_bom_list, dim=0)
    label_mclass_all = torch.cat(label_mclass_list, dim=0)
    recall_mclass = classk_metric(output_bom_all, label_mclass_all, -1, input_type='torch')[metric_idx]
    if visualize:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions, val_imgs
    else:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions

def val_no_cl(val_loader, model, criterion_bos, criterion_bom, gpu_exists,  criterion_cl, output_type, visualize=False, use_amp=False, selection_score='recall', ):

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

    for batch_id, (img1, img2, label1, label2, binary_y, position) in enumerate(val_loader):

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
                output_bos, output_bom = model(img1, img2)
                # label_binary = (label1 == label2).type(torch.float32).reshape([-1,1])
                if "Pre" not in output_type:
                    if output_bos is not None:
                        loss = (criterion_bos(output_bos, label_binary) + criterion_bom(output_bom, label2)) / 2
                    else:
                        loss = criterion_bom(output_bom, label2)
                else:
                    loss = criterion_cl(output_bos, output_bom, label_binary) 

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
            val_label_positions.append([output_bos_np, output_bom.detach().cpu().numpy(),
                                        label1, label2, label_binary, position])
        else:
            val_label_positions.append([label1, label2, label_binary, position])

    if output_bos is not None:
        label_binary_all = torch.cat(label_binary_list, dim=0)
        output_bos_all = torch.cat(output_bos_list, dim=0)
        recall_bi = classk_metric(output_bos_all, label_binary_all, 1, input_type='torch')[metric_idx]
    else:
        recall_bi = 0

    output_bom_all = torch.cat(output_bom_list, dim=0)
    label_mclass_all = torch.cat(label_mclass_list, dim=0)
    recall_mclass = classk_metric(output_bom_all, label_mclass_all, -1, input_type='torch')[metric_idx]
    if visualize:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions, val_imgs
    else:
        return losses.avg, accuracies_binary.avg, accuracies_mclass.avg, recall_bi, recall_mclass, val_label_positions

def val_cl_pre(val_loader, model, criterion_bos, criterion_bom, gpu_exists,  criterion_cl, output_type, visualize=False, use_amp=False, selection_score='recall', ):

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

    for batch_id, (img1, img2, label1, label2, binary_y, position) in enumerate(val_loader):

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
                output_bos, output_bom = model(img1, img2)
                # label_binary = (label1 == label2).type(torch.float32).reshape([-1,1])

                loss = criterion_cl(output_bos, output_bom, label_binary) 

        batch_size = img1.size(0)

        # update record
        losses.update(loss.item(), batch_size)

    return losses.avg

def main(args):
    date = args.date
    # set data path and set-up configs
    img_folder = args.data
    ckp_path = args.ckp
    region = args.region
    smooth = args.smooth
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
    save_checkpoint = True
    verbose_frequency = args.verbose_frequency
    optimizer_type = args.optimizer_type
    lr_schedule = args.lr_schedule
    lr_step_size = args.lr_step_size
    lr_gamma = args.lr_gamma
    jitter = args.jitter
    run_mode = args.mode
    rs_img_size = args.resize
    weight_decay = args.weight_decay
    version_name = args.version_name
    label_confidence = args.label_conf
    backbone_arch = args.arch.split('_')[0]

    # test_annotation_filename = os.path.join(img_folder, 'merged_annotation', date, f'annotation_labels_{region}_{date}.csv')
    img_folder_mnt = f'./results/{version_name}_csv'
    os.makedirs(img_folder_mnt, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    # specify the defect types, hyperparameters and dataset csvs for different component parts (ROI)
    if region == 'component':
        annotation_filename = os.path.join(img_folder, 'merged_annotation', date, f'train_labels_{region}resorted.csv')
        val_annotation_filename = os.path.join(img_folder, 'merged_annotation', date, f'val_labels_{region}resorted.csv')
        test_annotation_filename = os.path.join(img_folder, 'merged_annotation', date,
                                                f'bbtest_labels_{region}resorted.csv')
        
        # aug_train_pair_data_filename = None
        aug_train_pair_data_filenames = [
                                        os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_{region}_mergedresorted.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_{region}_240329c_finalresortedv2.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_240329c_finalresortedv2_update_241124.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_{region}_240410debug_final.csv'),

                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_trainpair_labels_{region}_240426syncfly_final.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_{region}_240428_final.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_selfpair_labels_{region}_240428_final_white.csv'),
                                        #  os.path.join(img_folder, 'merged_annotation', date,
                                        #               f'aug_train_pair_labels_{region}_240918_final_whitev2.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_240918_final_whitev2_update_241124.csv'),
                                                      
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_{region}_240930_final_white.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_241106_final_white.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_241118_final_white_update_241124.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_241217_final_white_DA561_2.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_241217_final_white_DA561_down_unkown.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_241223_final_white_DA574.csv'),
                                        os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_250116_final_white_DA662_dropoldpairs.csv'),
                                        os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_250121_final_white_DA678_dropoldpairs.csv'),
                                        # cross pair用于训练，原始数据作为测试
                                        os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_250319_final_white_DA796_797_dropoldpairs_all_cross_pair_ok.csv'),
                                        os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_250319_final_white_DA796_797_dropoldpairs_all_cross_pair_ng.csv'),
                                                    
                                        ]



        aug_val_pair_data_filenames = [
            os.path.join(img_folder, 'merged_annotation', date, f'aug_val_pair_labels_{region}_mergedresorted.csv'),
            # os.path.join(img_folder, 'merged_annotation', date, f'aug_val_pair_labels_{region}_240329c_finalresortedv2.csv'),
            # os.path.join(img_folder, 'merged_annotation', date, f'aug_test_pair_labels_{region}_240329c_finalresorted.csv'),
            os.path.join(img_folder, 'merged_annotation', date, f'aug_val_pair_labels_component_240329c_finalresortedv2_update_241124.csv'),
            os.path.join(img_folder, 'merged_annotation', date, f'aug_test_pair_labels_component_240329c_finalresorted_update_241124.csv'),

            os.path.join(img_folder, 'merged_annotation', date, f'aug_val_pair_labels_{region}_240410debug_final.csv'),
            os.path.join(img_folder, 'merged_annotation', date, f'aug_testpair_labels_{region}_240426syncfly_final.csv'),
            os.path.join(img_folder, 'merged_annotation', date, f'aug_val_pair_labels_{region}_240428_final.csv'),
            os.path.join(img_folder, 'merged_annotation', date, f'aug_test_pair_labels_{region}_240930_final_white.csv'),
            # os.path.join(img_folder, 'merged_annotation', date, f'aug_test_pair_labels_component_241118_final_white.csv'),
            os.path.join(img_folder, 'merged_annotation', date, f'aug_test_pair_labels_component_241118_final_white_update_241124.csv'),

            os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_test_pair_labels_component_241217_final_white_DA561_2.csv'),
            os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_test_pair_labels_component_241217_final_white_DA561_down_unkown.csv'),
            os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_test_pair_labels_component_241223_final_white_DA574.csv'),
            os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_test_pair_labels_component_250116_final_white_DA662_dropoldpairs.csv'),
            os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_test_pair_labels_component_250121_final_white_DA678_dropoldpairs.csv'),
            os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_test_pair_labels_component_250319_final_white_DA796_797_dropoldpairs.csv'),                                                    
            os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_component_250319_final_white_DA796_797_dropoldpairs.csv'),  
        ]
    
        if 'white' in version_name:
            print(version_name)
            rs_factor = 1.14
            rs_same_offset = 5
            random_rotate_angle = [0, 0, 0, 90, 180, 270]
            random_rotate_angle_sync = [0, 90, 180, 270]
            colorjitter = [0.1, 0.1, 0.15, 0.1]
            colorjitter_same = [0.1, 0.1, 0.15, 0.1]
            colorjitter_sync = [jitter, jitter, jitter, jitter/2]
            if 'fly' in version_name:
                defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11, 'fly': 12}

            else:
                defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}
            print(version_name, defect_code)

        else:
            rs_factor = 1.05
            rs_same_offset = 5
            random_rotate_angle = [0,180]
            random_rotate_angle_sync = [0, 90, 180, 270]
            colorjitter = [0.1, 0.1, 0.15, 0.1]
            colorjitter_same = [0.1, 0.1, 0.15, 0.1]
            colorjitter_sync = [jitter, jitter, jitter, jitter/2]
            defect_code = {'ok': 0, 'missing': 1, 'misaligned': 2, 'tombstone': 8, 'others': 11}

    elif region == 'body':
        annotation_filename = os.path.join(img_folder, 'merged_annotation', date, f'train_labels_{region}resorted.csv')
        val_annotation_filename = os.path.join(img_folder, 'merged_annotation', date, f'val_labels_{region}resorted.csv')
        test_annotation_filename = os.path.join(img_folder, 'merged_annotation', date,
                                                f'bbtest_labels_{region}resorted.csv')

        if 'white' in version_name:
            print(version_name)
            rs_factor = 1.14
            rs_same_offset = 10
            random_rotate_angle = [0, 0, 0, 180]
            random_rotate_angle_sync = [0, 90, 180, 270]
            colorjitter = [0.1, 0.1, 0.15, 0.1]
            colorjitter_same = [0.1, 0.1, 0.15, 0.1]
            colorjitter_sync = [jitter, jitter, jitter, jitter/2]
            if 'slim' in version_name:
                # defect_code = {'ok': 0, 'missing': 1, 'wrong': 3}
                defect_code = {'ok': 0, 'wrong': 3}
            else:
                defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11}

            aug_train_pair_data_filenames = [os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_{region}_mergedresorted.csv'),
                                         os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_train_pair_labels_{region}_240328b_finalresorted.csv'),
                                             os.path.join(img_folder, 'merged_annotation', date,
                                                          f'aug_train_selfpair_labels_{region}_240328b_finalresorted.csv'),
                                             os.path.join(img_folder, 'merged_annotation', date,
                                                        f'aug_train_pair_labels_{region}_merged_withcpmlresorted.csv'),
                                             os.path.join(img_folder, 'merged_annotation', date,
                                                          f'aug_train_pair_labels_{region}_240725_final_white.csv'),
#                                             os.path.join(img_folder, 'merged_annotation', date,
#                                                          f'aug_train_pair_labels_{region}_240913_final_white.csv'),
                                             os.path.join(img_folder, 'merged_annotation', date,
                                                         f'aug_train_pair_labels_{region}_240918_final_white.csv'),
                                             os.path.join(img_folder, 'merged_annotation', date,
                                                         f'aug_train_pair_labels_{region}_240919_final_white.csv'),
                                             os.path.join(img_folder, 'merged_annotation', date,
                                                         f'aug_train_pair_labels_{region}_240926sub_final_white.csv'),
                                             
            ]
            
            if 'newonly' not in version_name:
                aug_train_pair_data_filenames.append(os.path.join(img_folder, 'merged_annotation', date,
                                                         f'aug_train_pair_labels_{region}_240913_final_white.csv'))

            aug_val_pair_data_filenames = [os.path.join(img_folder, 'merged_annotation', date, f'aug_val_pair_labels_{region}_mergedresorted.csv'),
            os.path.join(img_folder, 'merged_annotation', date, f'aug_val_pair_labels_{region}_240328b_finalresorted.csv'),
            os.path.join(img_folder, 'merged_annotation', date, f'aug_test_pair_labels_{region}_240328b_finalresorted.csv'),
                                           os.path.join(img_folder, 'merged_annotation', date,
                                                      f'aug_val_pair_labels_{region}_merged_withcpmlresorted.csv'),
                                           os.path.join(img_folder, 'merged_annotation', date,
                                                         f'aug_test_pair_labels_{region}_240918sub_final_white.csv'),
                                           os.path.join(img_folder, 'merged_annotation', date,
                                                         f'aug_test_pair_labels_{region}_240919sub_final_white.csv'),
#             os.path.join(img_folder, 'merged_annotation', date, f'aug_test_pair_labels_{region}_240819_final_white.csv')
                                           ]

        else:
            rs_factor = 1.05
            rs_same_offset = 5
            random_rotate_angle = [0, 180]
            random_rotate_angle_sync = [0, 90, 180, 270]
            colorjitter = [0.1, 0.1, 0.15, 0.1]
            colorjitter_same = [0.1, 0.1, 0.15, 0.1]
            colorjitter_sync = [jitter, jitter, jitter, jitter/2]

            defect_code = {'ok': 0, 'wrong': 3}
    print(annotation_filename, val_annotation_filename, aug_train_pair_data_filenames, aug_val_pair_data_filenames)
    # defect class maps
    n_class = len(defect_code)
    defect_code_link = {v: i for i, v in enumerate(defect_code.values())}
    defect_class_considered = list(defect_code_link.keys())
    # defect_code = {'ok': 0, 'missing': 1, 'wrong': 3, 'tombstone': 8, 'others': 11, 'fly': 12}
    # whether to use mixed precision training
    if args.amp_lvl == 'f16':
        use_amp = True
    else:
        use_amp = False
    # whether to use weighted loss
    if 'wl' in version_name:
        weighted_loss = True
    else:
        weighted_loss = False

    # get backbone name and specify checkpoint path
    if 'pretrained' in args.arch:
        pretrained = True
    else:
        pretrained = False

    best_checkpoint_path = os.path.join(ckp_path,
                                        f'{region}{args.arch}rs{rs_img_size}s{seed}c{n_class}val{val_ratio}_ckp_best{version_name}{gamma}{smooth}{args.amp_lvl}j{jitter}lr{init_lr}nb{n_unit_binary}nm{n_unit_mclass}{output_type}top0.pth.tar')

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
    train_annotation_df, val_annotation_df = stratified_train_val_split(annotation_filename, val_ratio=val_ratio,
                                                                        defect_code=defect_code_link, region=region,
                                                                        groupby=f'defect_label', seed=seed,
                                                                        label_confidence = label_confidence,
                                                                        verbose=True)
    transform_mean = [0.485, 0.456, 0.406]
    transform_std = [0.229, 0.224, 0.225]
    if run_mode.startswith('train'):
        # training mode
        if run_mode == 'train_ft':
            Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = [], [], [], []
        else:
            n_max_pairs = 15

            # perform weighted resampling on training data
            Xtrain_resampled, ytrain_resampled, ybinary_resampled, material_train_resampled = mp_weighted_resampler(
                train_annotation_df, k=k_factor, batch_size=batch_size, n_max_pairs=n_max_pairs)

        if aug_train_pair_data_filenames is not None:
            # add in additional paired data for training
            aug_train_pair_data_df_raw = pd.concat([pd.read_csv(f, index_col=0) for f in aug_train_pair_data_filenames]).reset_index(drop=True)

            if label_confidence == 'certain':
                aug_train_pair_data_df_certain = aug_train_pair_data_df_raw[
                    (aug_train_pair_data_df_raw['confidence'] == 'certain') | (
                                aug_train_pair_data_df_raw['confidence'] == 'unchecked')].copy()
                print(f'aiug train pair certain = {len(aug_train_pair_data_df_certain)} out of {len(aug_train_pair_data_df_raw)}')
                aug_train_pair_data_df_raw = aug_train_pair_data_df_certain
            aug_train_pair_data_df = aug_train_pair_data_df_raw[[y in defect_class_considered for
                                                             y in aug_train_pair_data_df_raw['insp_y_raw']]].copy().reset_index(drop=True)
            n_aug = len(aug_train_pair_data_df)
            print(f' aug train withh {n_aug} image pairs')
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
        print(f'n_train_pairs = {len(train_pair_data_df)}')
        train_pair_data_df.to_csv(os.path.join(img_folder_mnt, f'annotation_train_pair_labels_{region}.csv'))
        # construct training dataset
        train_dataset = ImageLoader3(img_folder, Xtrain_resampled, ytrain_resampled, ybinary_resampled,
                                    material_train_resampled,
                                    compression_p = args.compression_p, 
                                    p_range = args.p_range,
                                    select_p = args.select_p,
                                    transform=transforms.Compose([
                                        transforms.Resize((int(rs_img_size*rs_factor), int(rs_img_size*rs_factor))),
                                        transforms.RandomCrop(rs_img_size),
                                        DiscreteRotate(angles=random_rotate_angle),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ColorJitter(
                                            brightness=colorjitter[0], contrast=colorjitter[1],
                                            saturation=colorjitter[2], hue=colorjitter[3]
                                        )
                                        ]),
                                    transform_same=transforms.Compose([
                                        transforms.Resize((int(rs_img_size + rs_same_offset), int(rs_img_size + rs_same_offset))),
                                        transforms.RandomCrop(rs_img_size),
                                        transforms.ColorJitter(
                                            brightness=colorjitter_same[0], contrast=colorjitter_same[1],
                                            saturation=colorjitter_same[2], hue=colorjitter_same[3]
                                        )
                                    ]),
                                    transform_sync=transforms.Compose([
                                        DiscreteRotate(angles=random_rotate_angle_sync),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ColorJitter(
                                            brightness=colorjitter_sync[0], contrast=colorjitter_sync[1],
                                            saturation=colorjitter_sync[2], hue=colorjitter_sync[3]
                                        ),
                                        transforms.Normalize(mean=transform_mean,
                                                             std=transform_std)
                                    ]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=n_workers, pin_memory=True)
        n_train = len(train_annotation_df)

    # generate pair inputs for test data
    print('preapar test data')
    Xtest, ytest, ytestraw, ybinary_test, material_test, test_annotation_df = generate_test_pairs(
        test_annotation_filename, defect_code=defect_code_link, region=region, mode='ng_only', label_confidence = label_confidence)
    # test_annotation_df.to_csv(os.path.join(img_folder, 'merged_annotation', date, f'test_image_pairs_{region}.csv')

    if args.if_cp_val:
        cp_ratio = args.compression_p
    else:
        cp_ratio = 0
    test_dataset = ImageLoader3(img_folder, Xtest, ytest, ybinary_test, material_test,
                                compression_p = cp_ratio, 
                                p_range = args.p_range,
                                select_p = args.select_p,
                               transform=transforms.Compose([
                                   transforms.Resize((rs_img_size, rs_img_size)),
                                   transforms.Normalize(mean=transform_mean,
                                                        std=transform_std),
                                   ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=(batch_size * 2),
                                              shuffle=False, num_workers=n_workers, pin_memory=True)
    n_test = len(test_annotation_df)
    print(f'n_val_pairs = {len(test_annotation_df)}')

    if val_ratio == 0:

        print('preapar val data')
        # generate pair inputs for validation data
        Xval, yval, yvalraw, ybinary_val, material_val, _ = generate_test_pairs(
            val_annotation_filename, defect_code=defect_code_link, region=region, mode='flexible', label_confidence = label_confidence)
        n_val = 0

        if aug_val_pair_data_filenames is not None:
            # add in additional paired data for validation
            aug_val_pair_data_df_raw = pd.concat([pd.read_csv(f, index_col=0) for f in aug_val_pair_data_filenames]).reset_index(drop=True)
            if label_confidence == 'certain':
                aug_val_pair_data_df_certain = aug_val_pair_data_df_raw[
                    (aug_val_pair_data_df_raw['confidence'] == 'certain') | (
                                aug_val_pair_data_df_raw['confidence'] == 'unchecked')].copy()
                print(f'aug val pair certain = {len(aug_val_pair_data_df_certain)} out of {len(aug_val_pair_data_df_raw)}')
                aug_val_pair_data_df_raw = aug_val_pair_data_df_certain
            aug_val_pair_data_df = aug_val_pair_data_df_raw[[y in defect_class_considered for
                                                             y in aug_val_pair_data_df_raw['insp_y_raw']]].copy().reset_index(drop=True)
            n_aug_val = len(aug_val_pair_data_df)
            print(f' aug val withh {n_aug_val} image pairs')
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
        val_pair_data_df.to_csv(os.path.join(img_folder_mnt, f'annotation_val_pair_labels_{region}.csv'))
        print(f'n_val_pairs = {len(val_pair_data_df)}')

    else:
        n_val = len(val_annotation_df)
        Xval, yval, yvalraw, ybinary_val, material_val = generate_val_pairs(train_annotation_df, val_annotation_df)
        val_pair_data = [x + yval[i] + yvalraw[i] + [ybinary_val[i], material_val[i]] for i, x in enumerate(Xval)]
        val_pair_data_df = pd.DataFrame(val_pair_data,
                                   columns=['ref_image', 'insp_image', 'ref_y', 'insp_y', 'ref_y_raw', 'insp_y_raw',
                                            'binary_y', 'material_id'])
        val_pair_data_df.to_csv(os.path.join(img_folder_mnt, f'annotation_val_pair_labels_{region}.csv'))
        print(f'n_val_pairs = {len(val_pair_data_df)}')


    val_dataset = ImageLoader3(img_folder, Xval, yval, ybinary_val, material_val,
                               compression_p = cp_ratio, 
                               p_range = args.p_range,
                               select_p = args.select_p,
                               transform=transforms.Compose([
                                  transforms.Resize((rs_img_size, rs_img_size)),
                                  transforms.Normalize(mean=transform_mean,
                                                       std=transform_std)
                              ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=(batch_size * 2),
                                             shuffle=False, num_workers=n_workers, pin_memory=True)

    print(f'n_train={n_train}, n_val={n_val}, n_test={n_test}, n_total={n_train + n_val}')

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
    if 'cbam' in backbone_arch and 'CL' not in output_type:
        print(backbone_arch)
        from models.MPB3_attention import MPB3net
        attention_list_str = backbone_arch.split('cbam')[-1]
        if len(attention_list_str.split('cbam')[-1]) == 0:
            attention_list = None
        else:
            attention_list = [int(attention_list_str)]
        print(attention_list)
        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                        output_form=output_type, attention_list=attention_list)
    # define classifier model
    elif 'cbam' in backbone_arch and 'CL' in output_type:
        print(backbone_arch)
        if 'CLC' in output_type:
            from models.MPB3_attn_no_Avg_ConstLearning import MPB3net  
        else:
            from models.MPB3_attn_ConstLearning import MPB3net
        attention_list_str = backbone_arch.split('cbam')[-1]
        if len(attention_list_str.split('cbam')[-1]) == 0:
            attention_list = None
        else:
            attention_list = [int(attention_list_str)]
        print(attention_list)
        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                        output_form=output_type, attention_list=attention_list)
    elif 'cbam' not in backbone_arch and 'CL' in output_type:
        if 'CLC' in output_type:
            from models.MPB3_no_Avg_ConstLearning import MPB3net
        else:
            from models.MPB3_ConstLearning import MPB3net
        model = MPB3net(backbone=backbone_arch, pretrained=pretrained, n_class=n_class, n_units=n_units, output_form=output_type)

    else:
        from models.MPB3 import MPB3net

        model = MPB3net(backbone=backbone_arch, pretrained=False, n_class=n_class, n_units=n_units,
                        output_form=output_type)

    n_params = sum([p.numel() for p in model.parameters()])
    print(f'model has {n_params / (1024 * 1024)} M params')

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

    # refix random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
                model.load_state_dict(loaded_state_dict, strict=False)
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
            if 'CL' in output_type and 'no' not in output_type and 'Pre' not in output_type:
                val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions, val_imgs = val(
                    val_loader, model, criterion_bos, criterion_bom, gpu_exists, visualize=True, use_amp=use_amp, selection_score=selection_score)

                test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = val(
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
            elif 'CL' in output_type and 'Pre' in output_type:
                val_losses = val_cl_pre(val_loader, model, criterion_bos, criterion_bom, gpu_exists, cl_loss, output_type, visualize=True, use_amp=use_amp, selection_score=selection_score)
                print(f'Before finetune: val_losses={val_losses}')
            else:

                val_losses, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions, val_imgs = val_no_cl(
                    val_loader, model, criterion_bos, criterion_bom, gpu_exists, cl_loss, output_type, visualize=True, use_amp=use_amp, selection_score=selection_score)

                test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = val_no_cl(
                    test_loader,
                    model,
                    criterion_bos,
                    criterion_bom,
                    gpu_exists, cl_loss, output_type,use_amp=use_amp, selection_score=selection_score)
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

        for epoch in range(start_epoch, epochs):

            # train the model for 1 epoch
            epoch_time_start = time.time()
            if 'CL' in output_type and 'no' not in output_type and 'Pre' not in output_type:
                train_loss, train_acc = train(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists,
                                          use_amp, scaler)
            elif 'Pre' in output_type:
                train_loss = train_cl_pre(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists,
                                          use_amp, scaler)
                train_acc = 0
            else:
                train_loss, train_acc = train_no_cl(train_loader, model, criterion_bos, criterion_bom, cl_loss, optimizer, gpu_exists,
                                          use_amp, scaler)
            epoch_time = time.time() - epoch_time_start
            scheduler.step()
            # validate the model
            if 'CL' in output_type and 'no' not in output_type and 'Pre' not in output_type:
                val_loss, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions = val(val_loader,
                                                                                                              model,
                                                                                                              criterion_bos,
                                                                                                              criterion_bom,
                                                                                                              gpu_exists,
                                                                                                              use_amp=use_amp,
                                                                                                              selection_score=selection_score)

                test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = val(
                    test_loader, model, criterion_bos, criterion_bom, gpu_exists, use_amp=use_amp, selection_score=selection_score)
            
            elif 'Pre' in output_type:
                val_loss = val_cl_pre(val_loader, model, criterion_bos, criterion_bom, gpu_exists, cl_loss, output_type, use_amp=use_amp, selection_score=selection_score)
                print(f'Before finetune: val_losses={val_loss}')
                val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions = 0,0,0,0,0
                test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = 0,0,0,0,0,0
            else:
                val_loss, val_acc_bi, val_acc_mc, val_recall_bi, val_recall_mclass, val_label_positions = val_no_cl(val_loader,
                                                                                                            model,
                                                                                                            criterion_bos,
                                                                                                            criterion_bom,
                                                                                                            gpu_exists, cl_loss, output_type,
                                                                                                            use_amp=use_amp,
                                                                                                            selection_score=selection_score)

                test_loss, test_acc_bi, test_acc_mc, test_recall_bi, test_recall_mclass, test_label_positions = val_no_cl(
                    test_loader, model, criterion_bos, criterion_bom, gpu_exists, cl_loss, output_type, use_amp=use_amp, selection_score=selection_score)              
            # record loss and other metrics
            epoch_train_losses.append(train_loss)
            if 'Pre' in output_type:
                epoch_train_accuracies.append(0)
                epoch_val_accuracies_mclass.append(0)
            else:
                epoch_train_accuracies.append(train_acc.cpu().numpy())
                epoch_val_accuracies_mclass.append(val_acc_mc.cpu().numpy())

            epoch_val_losses.append(val_loss)
            if output_type == 'mclass' or 'Pre' in output_type:
                epoch_val_accuracies_binary.append(0.0)
            else:
                epoch_val_accuracies_binary.append(val_acc_bi.cpu().numpy())

            if epoch % verbose_frequency == 0:
                print(f'Ep {epoch}: train loss={train_loss:.3f}, train acc={train_acc:.3f}%,'
                    f'val loss={val_loss:.3f}, mclass val acc={val_acc_mc:.3f}%, '
                    f'binary val acc={val_acc_bi:.3f}%,  binary recall={val_recall_bi:.3f}%, mclass recall={val_recall_mclass:.3f}%||'
                    f'binary test acc={test_acc_bi:.3f}%, mclass test acc={test_acc_mc:.3f}%, '
                    f'binary test recall={test_recall_bi:.3f}%, mclass test recall={test_recall_mclass:.3f}%, '
                    f'time={epoch_time:.3f} s')
                
            if 'Pre' not in output_type:
                current_metric_score = val_acc_bi + val_acc_mc + (val_recall_bi + val_recall_mclass) * 150
            else:
                current_metric_score = 1/abs(val_loss)

            if (save_checkpoint and (run_mode == 'train' or run_mode == 'train_resume')):
                # save the top 3 checkpts
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

                # save train and val statistics for visualization and checking
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
    parser.add_argument('--data', default='/home/robinru/shiyuan_projects/smt_defect_data_pipeline/data_white', type=str,
    # parser.add_argument('--data', default='/home/classification/data_clean', type=str,
                        help='path to image datasets and annotation labels')
    parser.add_argument('--ckp', default='./models/checkpoints', type=str,
                        help='path to save and load checkpoint')
    parser.add_argument('--region', default='body', type=str,
                        help='component instance region data to use: component, body')
    parser.add_argument('--arch', default='fcdropoutmobilenetv3large', type=str,
                        help='backbone arch to be used：fcdropoutresnet18, fcdropoutmobilenetv3large, resnet18')
    parser.add_argument('--date', default='231013_white', type=str,
                        help='data processing date ')
    parser.add_argument('--version_name', default='vwhite', type=str,
                        help='model version name')
    parser.add_argument('--resume',
                        default=None,
                        # default='./models/checkpoints/v2.81best/pinsfcdropoutmobilenetv3large_pretrainedrs224s42c7val0.0_ckp_bestv2.720.0f16j0.4lr0.1nb256nm128dual2top0.pth.tar',
                        type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--reload_mode',
                        default='full',
                        type=str, help='which mode to load the ckp: full, backbone, skip_mismatch')
    parser.add_argument('--mode', default='train', type=str,
                        help='running mode: validation or training')
    parser.add_argument('--output_type', default='dual2', type=str,
                        help='specify the output tyep: dual, dual2, mclass')
    parser.add_argument('--score', default='recall', type=str,
                        help='selection score: recall,  f1')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--k', default=2.5, type=float,
                        help='weighted sampling multiplier')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--resize', default=224, type=int,
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
    parser.add_argument('-bs', '--batch_size', default=64, type=int,
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
    parser.add_argument('--optimizer_type', default='sgdft', type=str,
                        help='optimizer to use: SGD or RMSProp')
    parser.add_argument('--lr_schedule', default='cosine', type=str,
                        help='learning rate scheduling: cosine or step decay')
    
    parser.add_argument('--lr_schedule_T', default=600, type=int,
                        help='learning rate scheduling: cosine or step decay')
    parser.add_argument('--lr_step_size', default=7, type=float,
                        help='step decay size')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='step decay gamma')
    parser.add_argument('-vr', '--val_ratio', default=0.0, type=float,
                        help='ratio of validation ')
    parser.add_argument('-vf', '--verbose_frequency', default=1, type=int,
                        help='frequency to print results')
    parser.add_argument('--amp_lvl', default='f16', type=str,
                        help='precision level used:f32, f16')
    parser.add_argument('--label_conf', default='all', type=str,
                        help='label confidence: all, certain')
    
    parser.add_argument('-cld', '--constrative_loss_dist', default='Cosin', type=str,
                        help='the calculation function of constrative learning distance')
    parser.add_argument('--compression_p', default=0, type=float,
                        help='图片压缩概率')   
    parser.add_argument('--p_range', default=[65, 95], type=float,
                        help='图片压缩程度, 0-100, 值越小压缩程度越大')
    parser.add_argument('--if_cp_val', default=0, type=int,
                        help='验证集是否进行压缩', choices=[0, 1])     
    # 实验证明PIL的压缩会导致一定程度的失真，所以最好为0，用opencv的压缩;
    parser.add_argument('--select_p', default=0, type=float,
                        help='选择opencv压缩还是PIL压缩, select_p=0为opencv压缩, select_p=1为PIL压缩,0-1之间则随机选择, None则随机选择')
    
    args = parser.parse_args()
    print(args)
    # run experiment
    val_loader, model = main(args)