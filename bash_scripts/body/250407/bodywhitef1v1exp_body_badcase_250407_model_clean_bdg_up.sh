#!/usr/bin/env bash

set -x
set -e

GPU_COUNT=`nvidia-smi -L | wc -l`

DATE=`date '+%F'`

WORKDIR="${WORKDIR:=/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification}"
python3 ${WORKDIR}/main_advanced_cl_model_clean2.py --arch=fcdropoutmobilenetv3large --region=body --output_type=dual2 -lr=0.025 --version_name=v2.11.0whiteslimf1certainmdbdg --epochs=600 --gpu=0 -nb=256 -nm=256 --k=3 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/body/v2.11/bodywhiteslimf1v2.11.0mdbdg" --data="/dev/shm/classification/data_clean" --date="2411" --worker=12 --batch_size=256 --score='f1' --label_conf='certain' --resume="${WORKDIR}/models/checkpoints/body/v2.10tp1/bodywhiteslimf1v2.10.2select/bodyfcdropoutmobilenetv3largers224s42c2val0.0_ckp_bestv2.10.2whiteslimf1certainbdg20.0f16j0.4lr0.025nb256nm256dual2last.pth.tar" --resize=224 