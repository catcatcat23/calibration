#!/usr/bin/env bash

set -x 
set -e

GPU_COUNT=`nvidia-smi -L | wc -l`

DATE=`date '+%F'`

WORKDIR="${WORKDIR:=/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification}"

python3 ${WORKDIR}/main_advanced_kd_cl_tb_cp_random.py --arch=fcdropoutmobilenetv3large --output_type=dual2 --specila_issue_list=DA730,DA730-cross_pair --if_cp_val=0  --select_p=0.5 --compression_p=0.5 --region=singlepad -lr=0.025 --version_name=v0.11.1f1certaincp05crosspairDA730TG --epochs=600 --gpu=0 -nb=256 -nm=256 -ji=0.4 --resize_w=64 --resize_h=64 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/singlepadv0.11.1cp05crosspairDA730TG" --data="/dev/shm/classification/data_clean" --date="2501" --worker=18 --batch_size=256 -vr=0.1 --score='f1' --label_conf='certain' --reload_mode='skip_mismatch' -kdr=0.0 --resume="${WORKDIR}/models/checkpoints/singlepadv0.10impl/singlepadfcdropoutmobilenetv3largers6464s42c3val0.1b256_ckp_bestv0.10.3f1certainlut05cp0520.0j0.4lr0.025nb256nm256dual2last.pth.tar" --lut_p=0 &
python3 ${WORKDIR}/main_advanced_kd_cl_tb_cp_random.py --arch=fcdropoutmobilenetv3large --output_type=dual2 --specila_issue_list=DA730,DA730-cross_pair --if_cp_val=0  --select_p=0.5 --compression_p=0.5 --region=singlepad -lr=0.1 --version_name=v0.11.1f1certainretraincp05crosspairDA730TG --epochs=600 --gpu=0 -nb=256 -nm=256 -ji=0.4 --resize_w=64 --resize_h=64 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/singlepadv0.11.1retraincp05crosspairDA730TG" --data="/dev/shm/classification/data_clean" --date="2501" --worker=18 --batch_size=256 -vr=0.1 --score='f1' --label_conf='certain' --reload_mode='skip_mismatch' -kdr=0.0 --seed=8 --lut_p=0 &
# python3 ${WORKDIR}/main_advanced_kd_cl_tb_cp.py --arch=fcdropoutmobilenetv3large --output_type=dual2  --if_cp_val=1 --compression_p=0.1 --region=singlepad -lr=0.025 --version_name=v0.10.1f1certaincp01 --epochs=20 --gpu=0 -nb=256 -nm=256 -ji=0.4 --resize_w=64 --resize_h=64 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/singlepadv0.10.1cp01" --data="/dev/shm/classification/data_clean" --date="2501" --worker=18 --batch_size=256 -vr=0.1 --score='f1' --label_conf='certain' --reload_mode='skip_mismatch' -kdr=0.0 --resume="${WORKDIR}/models/checkpoints/singlepadv0.9select/singlepadfcdropoutmobilenetv3largers6464s42c3val0.1b256_ckp_bestv0.9.0f1certainlut0520.0j0.4lr0.025nb256nm256dual2last.pth.tar" --lut_p=0 &