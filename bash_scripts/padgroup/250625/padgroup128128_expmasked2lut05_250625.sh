#!/usr/bin/env bash

set -x 
set -e

GPU_COUNT=`nvidia-smi -L | wc -l`

DATE=`date '+%F'`

WORKDIR="${WORKDIR:=/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification}"

python3 ${WORKDIR}/main_advanced_kd_cl_tb_cp_random.py --arch=fcdropoutmobilenetv3large --if_save_test_best_model=True --region=padgroup -lr=0.025 --version_name=v1.22.0slimmaskedf1certainrgbwhitelut05 --epochs=600 --gpu=0 -nb=128 -nm=128 -ji=0.4 --resize_w=128 --resize_h=128 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/padgroup/v1.22/padgroupv1.22.0rgbwhitelut05" --data="/dev/shm/classification/data_clean" --date="2501" -vr=0.1 --worker=18 --batch_size=256 --score='f1' -kdr=0.0 --label_conf='certain' --reload_mode='skip_mismatch' --resume="${WORKDIR}/models/checkpoints/padgroup/v1.21_jxlzy_tp1/padgroupv1.21.2tp1select/padgroupfcdropoutmobilenetv3largers128128s42c2val0.1b256_ckp_bestv1.21.2slimmaskedf1certainlut05edtg20.0j0.4lr0.025nb128nm128dual2bestacc.pth.tar" -lutp=0.5 &

