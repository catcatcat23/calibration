#!/usr/bin/env bash

set -x
set -e

GPU_COUNT=`nvidia-smi -L | wc -l`

DATE=`date '+%F'`

WORKDIR="${WORKDIR:=/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification}"
python3 ${WORKDIR}/main_advanced_cl_model_clean2.py --arch=fcdropoutmobilenetv3large --if_save_test_best_model=True --higher_threshold=0.5 --defect_dirs=NONE,WRONG,OVER_SOLDER,CONCAT,WORD_PAIR,SEHUAN_OK,SEHUAN_NG --region=body --output_type=dual2 -lr=0.025 --version_name=v2.15.0whiteslimf1certainmdbdg --epochs=600 --gpu=0 -nb=256 -nm=256 --k=3 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/body/v2.15.0/bodywhiteslimf1v2.15.0mdbdg" --data="/dev/shm/classification/data_clean" --date="2411" --worker=12 --batch_size=256 --score='f1' --label_conf='certain' --resume="${WORKDIR}/models/checkpoints/body/v2.14_zts_new1/bodywhiteslimf1v2.14ztsfinalselect/bodyfcdropoutmobilenetv3largers224s42c2val0.0_ckp_bestv2.14.3whiteslimf1certainmdbdgadd2d20.0f16j0.4lr0.025nb256nm256dual2bestbiacc.pth.tar" --resize=224 