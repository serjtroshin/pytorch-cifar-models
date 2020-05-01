#!/bin/bash

# forward

work_dir=DEQsequential-pretraining-grid-exp-longtime

# for optim in adam
# do
# for lr in 0.01 0.001 0.1
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

for optim in adam
do
for lr in 0.01 0.001
do
for mode in ''
do
    prelr=$lr
    sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 10 \
    --name exp.$mode.$lr.$prelr.track \
    --optimizer $optim \
    --lr $lr \
    --pretrain_steps 0 \
    --test_mode broyden \
    --inplanes 64 \
    --resume pretrained_models/pretrained_5layer_oneblock64.$optim.$prelr/checkpoint.pth \
    --work_dir experiments/$work_dir $mode \
    --track_running_stats
done
done
done

# tensorboard dev upload --logdir experiments/$work_dir-10

