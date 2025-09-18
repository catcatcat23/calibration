#!/usr/bin/env bash

set -x 
set -e

GPU_COUNT=`nvidia-smi -L | wc -l`

DATE=`date '+%F'`

WORKDIR="${WORKDIR:=/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification}"

python3 ${WORKDIR}/main_advanced_kd_cl_tb_cp_random_lut1.py --arch=fcdropoutmobilenetv3large --if_save_test_best_model=True --region=padgroup --specila_issue_list=sgjlc_op_rgb,xr_op_rgb,ry_jsd_rgb,szcc_lx_op -lr=0.025 --version_name=v1.22.6slimmaskedf1certainlut1edseed42 --epochs=600 --gpu=0 -nb=128 -nm=128 -ji=0.4 --resize_w=128 --resize_h=128 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/padgroup/v1.22.6lut1seed42/padgroupv1.22.6lut1seed42ed" --data="/dev/shm/classification/data_clean" --date="2501" -vr=0.1 --worker=18 --batch_size=256 --score='f1' -kdr=0.0 --label_conf='certain' --resume="${WORKDIR}/models/checkpoints/padgroup/padgroupv1.22.4/padgroupfcdropoutmobilenetv3largers128128s42c2val0.1b256_ckp_bestv1.22.4slimmaskedf1certainlut05ed20.0j0.4lr0.025nb128nm128dual2bestacc.pth.tar" -lutp=1 &
python3 ${WORKDIR}/main_advanced_kd_cl_tb_cp_random_lut1.py --arch=fcdropoutmobilenetv3large --if_save_test_best_model=True --region=padgroup --specila_issue_list=sgjlc_op_rgb,xr_op_rgb,ry_jsd_rgb,szcc_lx_op -lr=0.025 --version_name=v1.22.6slimmaskedf1certainretrainlut1edseed42 --epochs=600 --gpu=0 -nb=128 -nm=128 -ji=0.4 --resize_w=128 --resize_h=128 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/padgroup/v1.22.6lut1seed42/padgroupv1.22.6lut1retrainseed42ed" --data="/dev/shm/classification/data_clean" --date="2501" -vr=0.1 --worker=18 --batch_size=256 --score='f1' -kdr=0.0 --label_conf='certain' -lutp=1
