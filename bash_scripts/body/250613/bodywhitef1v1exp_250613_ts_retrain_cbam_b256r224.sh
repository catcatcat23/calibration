#!/usr/bin/env bash

set -x
set -e

GPU_COUNT=`nvidia-smi -L | wc -l`

DATE=`date '+%F'`

WORKDIR="${WORKDIR:=/mnt/pvc-nfs-dynamic/xianjianming/train_tasks/250123_smt_defect_classification}"
python3 ${WORKDIR}/main_advanced_cl_model_clean2.py --arch=fcdropoutmobilenetv3largecsmerge --reload_mode=skip_mismatch --if_save_test_best_model=True --higher_threshold=0.5 --defect_dirs=NONE,WRONG,OVER_SOLDER,CONCAT,WORD_PAIR,SEHUAN_OK,SEHUAN_NG --specila_issue_list=796797,hty_op --region=body --resize=224 --batch_size=256 --output_type=dual2 -lr=0.1 --version_name=v2.14.101cbamwhiteslimf1certaintsretrainr224 --epochs=600 --gpu=0 -nb=256 -nm=256 --k=3 --optimizer_type=sgd --ckp="${WORKDIR}/models/checkpoints/body/v2.101/bodywhiteslimf1v2.14.101cbamtsretrainr224" --data="/dev/shm/classification/data_clean" --date="2411" --worker=12 --score='f1' --label_conf='certain' 