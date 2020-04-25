#!/bin/bash

# forward

work_dir=DEQsequential-save-model

# sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 -ct 10 \
#     --name test_forward \
#     --pretrain_steps 300000 \
#     --test_mode forward \
#     --optimizer sgd --lr 0.1 \
#     --n_layer 18 --work_dir experiments/$work_dir

# for optim in sgd adam
for optim in adam
do
# for lr in 0.1 0.01 0.001
# for lr in 1 0.05
for lr in 0.01 0.001
do
    sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 -ct 10 \
    --name test_broyden \
    --optimizer $optim \
    --lr $lr \
    --pretrain_steps 0 \
    --test_mode broyden \
    --resume pretrained_models/checkpoint.pth \
    --n_layer 5 --work_dir experiments/$work_dir \
    --save_dir pretrained_models/$optim.$lr.oneblock
done
done

# tensorboard dev upload --logdir experiments/$work_dir-10

# sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
# --name test_FORWARD_shallow \
# --clip 50 --test_mode forward --n_layer 5 --work_dir experiments/$work_dir


# sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
# --name test_FORWARD_long \
# --clip 50 --test_mode forward --n_layer 18 --work_dir experiments/$work_dir

# sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
# --name test_BROYDEN --f_thres 30 --pretrain_steps 300 \
# --clip 50 --test_mode broyden --n_layer 18 --work_dir experiments/$work_dir

# sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
# --name test_BROYDEN --f_thres 30 --pretrain_steps 300 \
# --clip 50 --test_mode broyden --n_layer 5 --work_dir experiments/$work_dir

