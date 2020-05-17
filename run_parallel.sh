#!/bin/bash

# forward

work_dir=DEQparallel-baseline

for optim in sgd
do
for lr in 0.1
do
sbatch -c 2 -G 1  run.sh --epoch 160 --batch-size 128 -ct 10 \
    --name test_forward \
    --optimizer sgd --lr $lr \
    --inplanes 61 \
    --midplanes 61 \
    --model_type wtii_preact_parresnet110_cifar \
    --n_layer 18 --work_dir experiments/$work_dir
done
done

# for optim in adam
# do
# for lr in 0.01 0.1 0.001
# do
#     sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 10 \
#     --name $work_dir.f_thres15 \
#     --optimizer $optim \
#     --lr $lr \
#     --n_layer 5 \
#     --f_thres 15 \
#     --pretrain_steps 20 \
#     --test_mode broyden \
#     --inplanes 16 \
#     --midplanes 283 \
#     --track_running_stats \
#     --model_type deq_parresnet110_cifar \
#     --work_dir experiments/$work_dir \
#     --save_dir result/$work_dir.$name.$optim.$lr.61
# done
# done

tensorboard dev upload --logdir experiments/$work_dir-10

