#!/usr/bin/env bash

set -x 
set -e

GPU_COUNT=`nvidia-smi -L | wc -l`

DATE=`date '+%F'`

WORKDIR="${WORKDIR:=/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification}"

python3 ${WORKDIR}/main_advanced_fusion_white_rgb.py --arch=fcdropoutmobilenetv3largefusion --fusion_type=cat_gray --specila_issue_list=szcc_lx_op,njzh_rgbwhite --region=padgroup -lr=0.025 --version_name=v1.22.6slimmaskedf1certainrgbwhitelut05catgray --epochs=600 --gpu=0 -nb=128 -nm=128 -ji=0.4 --resize_w=128 --resize_h=128 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/padgroup/v1.22.6fusion/padgroupv1.22.6rgbwhitelut05catgray" --data="/dev/shm/classification/data_clean" --date="2501" -vr=0.1 --worker=18 --batch_size=256 --score='f1' -kdr=0.0 --label_conf='certain' --reload_mode='skip_mismatch' --resume="${WORKDIR}/models/checkpoints/padgroup/v1.22.5/padgroupv1.22.5selct/padgroupfcdropoutmobilenetv3largers128128s42c2val0.1b256_ckp_bestv1.22.5slimmaskedf1certainlut0520.0j0.4lr0.025nb128nm128dual2bestacc.pth.tar" -lutp=0.5 &

