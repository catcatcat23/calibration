#!/usr/bin/env bash

set -x 
set -e

GPU_COUNT=`nvidia-smi -L | wc -l`

DATE=`date '+%F'`

WORKDIR="${WORKDIR:=/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification}"


# python3 ${WORKDIR}/main_advanced_kd_v2.py --arch=mobilenetv3small --region=singlepinpad -lr=0.025 --version_name=v2.4.0f1certainNGonlylut --epochs=600 --gpu=0 -nb=256 -nm=256 -ji=0.4 --resize_w=128 --resize_h=32 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints" --data="/dev/shm/classification/data_clean" --date="240912" --reload_mode='full' --worker=8 --batch_size=256 -vr=0.15 --score='f1' --label_conf='certain' -kdr=0.0 --resume="${WORKDIR}/models/checkpoints/singlepinpadold/singlepinpadmobilenetv3smallrs12832s42c3val0.15b256_ckp_bestv2.0.1f1certainretrain20.0j0.4lr0.1nb256nm256dual2top1.pth.tar" --lut_p=1.0 &

# python3 ${WORKDIR}/main_advanced_kd_v2.py --arch=mobilenetv3small --region=singlepinpad -lr=0.025 --version_name=v2.4.1f1certainNGonlylut --epochs=600 --gpu=0 -nb=256 -nm=256 -ji=0.4 --resize_w=128 --resize_h=32 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints" --data="/dev/shm/classification/data_clean" --date="240912" --reload_mode='full' --worker=8 --batch_size=256 -vr=0.15 --score='f1' --label_conf='certain' -kdr=0.0 --resume="${WORKDIR}/models/checkpoints/singlepinpadold/singlepinpadmobilenetv3smallrs12832s42c3val0.15b256_ckp_bestv2.2.1f1certain20.0j0.4lr0.025nb256nm256dual2top1.pth.tar" --lut_p=1.0 &

# python3 ${WORKDIR}/main_advanced_kd_v2.py --arch=mobilenetv3small --region=singlepinpad -lr=0.1 --version_name=v2.4.0f1certainNGonlyretrainlut --epochs=600 --gpu=0 -nb=256 -nm=256 -ji=0.4 --resize_w=128 --resize_h=32 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints" --data="/dev/shm/classification/data_clean" --date="240912" --worker=8 --batch_size=256 -vr=0.15 --score='f1' --label_conf='certain' -kdr=0.0 --lut_p=1.0 

python3 ${WORKDIR}/main_advanced_kd_cl_tb_cp_random.py --arch=mobilenetv3small --output_type=dual2 --if_save_test_best_model=True --if_cp_val=0 --select_p=0.5 --compression_p=0.5 --region=singlepinpad -lr=0.025 --version_name=v2.16.4f1certaincp05 --epochs=600 --gpu=0 -nb=256 -nm=256 -ji=0.4 --resize_w=128 --resize_h=32 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/singlepinpad/v2.16.4/singlepinpadv2.16.4cp05" --data="/dev/shm/classification/data_clean" --date="2501" --reload_mode='skip_mismatch' --worker=8 --batch_size=256 -vr=0.15 --score='f1' --label_conf='certain' -kdr=0.0 --resume="${WORKDIR}/models/checkpoints/singlepinpad/v2.16.0/singlepinpadv2.16.0select/singlepinpadmobilenetv3smallrs12832s42c3val0.15b256_ckp_bestv2.16.0f1certainNGonlycp05clean20.0j0.4lr0.025nb256nm256dual2bestacc.pth.tar" --lut_p=0.0 &
python3 ${WORKDIR}/main_advanced_kd_cl_tb_cp_random.py --arch=mobilenetv3small --output_type=dual2 --if_save_test_best_model=True --if_cp_val=0 --select_p=0.5 --compression_p=0.5 --region=singlepinpad -lr=0.1 --version_name=v2.16.4f1certainretraincp05 --epochs=600 --gpu=0 -nb=256 -nm=256 -ji=0.4 --resize_w=128 --resize_h=32 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/singlepinpad/v2.16.4/singlepinpadv2.16.4retraincp05" --data="/dev/shm/classification/data_clean" --date="2501" --reload_mode='skip_mismatch' --worker=8 --batch_size=256 -vr=0.15 --score='f1' --label_conf='certain' -kdr=0.0 --lut_p=0 &
