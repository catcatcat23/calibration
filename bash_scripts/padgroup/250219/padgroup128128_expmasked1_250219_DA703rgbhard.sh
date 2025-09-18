#!/usr/bin/env bash

set -x 
set -e

GPU_COUNT=`nvidia-smi -L | wc -l`

DATE=`date '+%F'`

WORKDIR="${WORKDIR:=/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification}"

python3 ${WORKDIR}/main_advanced_kd_cl_tb.py --arch=fcdropoutmobilenetv3large --region=padgroup --specila_issue_list=DA728hardrgb -lr=0.025 --version_name=v1.16.1slimmaskedf1certainTG --epochs=600 --gpu=0 -nb=128 -nm=128 -ji=0.4 --resize_w=128 --resize_h=128 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/padgroup/v1.16/padgroupv1.16.1TG" --data="/dev/shm/classification/data_clean" --date="2501" -vr=0.1 --worker=18 --batch_size=256 --score='f1' -kdr=0.0 --label_conf='certain' --resume="${WORKDIR}/models/checkpoints/padgroup/v1.15/padgroup1.14impl/padgroupfcdropoutmobilenetv3largers128128s42c2val0.1b256_ckp_bestv1.14.0slimmaskedf1certaincp020.0j0.4lr0.025nb128nm128dual2top0.pth.tar" -lutp=0 &
python3 ${WORKDIR}/main_advanced_kd_cl_tb.py --arch=fcdropoutmobilenetv3large --region=padgroup --specila_issue_list=DA728hardrgb -lr=0.025 --version_name=v1.16.1slimmaskedf1certainNGonlyTG --epochs=600 --gpu=0 -nb=128 -nm=128 -ji=0.4 --resize_w=128 --resize_h=128 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/padgroup/v1.16/padgroupv1.16.1NGonlyTG" --data="/dev/shm/classification/data_clean" --date="2501" -vr=0.1 --worker=18 --batch_size=256 --score='f1' -kdr=0.0 --label_conf='certain' --resume="${WORKDIR}/models/checkpoints/padgroup/v1.15/padgroup1.14impl/padgroupfcdropoutmobilenetv3largers128128s42c2val0.1b256_ckp_bestv1.14.0slimmaskedf1certaincp020.0j0.4lr0.025nb128nm128dual2top0.pth.tar" -lutp=0 

