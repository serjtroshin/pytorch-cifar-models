#!/bin/bash

# forward

work_dir=DEQsequential-fullmodel-test

# sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
# --name test_FORWARD_shallow \
# --clip 50 --test_mode forward --n_layer 5 --work_dir experiments/$work_dir


# sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
# --name test_FORWARD_long \
# --clip 50 --test_mode forward --n_layer 18 --work_dir experiments/$work_dir

# sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
# --name test_BROYDEN --f_thres 30 --pretrain_steps 300 \
# --clip 50 --test_mode broyden --n_layer 18 --work_dir experiments/$work_dir

sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
--name test_BROYDEN --f_thres 30 --pretrain_steps 300 \
--clip 50 --test_mode broyden --n_layer 5 --work_dir experiments/$work_dir


# tensorboard dev upload --logdir experiments/'$work_dir'-10