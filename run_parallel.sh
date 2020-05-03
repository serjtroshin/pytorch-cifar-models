#!/bin/bash

# forward

work_dir=DEQsequential-pretraining-grid-exp-longtime

# for optim in adam
# do
# for lr in 0.001 0.1
# do
# sbatch -c 2 -G 1  run.sh --epoch 160 --batch-size 128 -ct 10 \
#     --name test_forward \
#     --pretrain_steps 300000 \
#     --test_mode forward \
#     --optimizer sgd --lr $lr \
#     --inplanes 64 \
#     --n_layer 18 --work_dir experiments/$work_dir
# done
# done

# add wnorm
for optim in adam
do
for lr in 0.01
do
    prelr=$lr
    bash run.sh --epoch 160 --batch-size 128 -ct 10 \
    --name exp.$mode.$lr \
    --optimizer $optim \
    --lr $lr \
    --pretrain_steps 20 \
    --test_mode broyden \
    --inplanes 16 \
    --track_running_stats \
    --work_dir experiments/$work_dir
done
done

# tensorboard dev upload --logdir experiments/$work_dir-10

