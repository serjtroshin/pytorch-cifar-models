#!/bin/bash

# forward

work_dir=DEQsequential-midplanes

# name=sequential_broyden
# for opt in adam
# do
# for lr in 0.1 0.01 0.001
# do
# sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 10 \
# --name $name \
# --optimizer $opt --lr $lr \
# --test_mode broyden \
# --n_layer 5 \
# --inplanes 61 \
# --pretrain_steps 20 \
# --work_dir experiments/$work_dir \
# --track_running_stats \
# --save_dir result/$work_dir.$name.$opt.$lr.61
# done
# done


name=sequential_broyden
for opt in adam
do
for lr in 0.01 0.001
do
for f_thres in 15 30
do
sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 10 \
--name $name.f_thres$f_thres \
--optimizer $opt --lr $lr \
--test_mode broyden \
--n_layer 5 \
--inplanes 16 \
--f_thres $f_thres \
--midplanes 190 \
--pretrain_steps 50 \
--model_type wtii_deq_preact_resnet110_cifar \
--work_dir experiments/$work_dir \
--track_running_stats \
--save_dir result/$work_dir.$name.$opt.$lr.inplanes16.midplanes190
done
done
done

# name=sequential_broyden
# for opt in adam
# do
# for lr in 0.001
# do
# sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 10 \
# --name $name \
# --optimizer $opt --lr $lr \
# --test_mode broyden \
# --n_layer 5 \
# --inplanes 61 \
# --midplanes 61 \
# --pretrain_steps 0 \
# --work_dir experiments/$work_dir \
# --track_running_stats \
# --model_type wtii_deq_preact_resnet110_cifar \
# --resume result/DEQsequential-3blocks.sequential_broyden.adam.0.001.61/checkpoint.pth \
# --save_dir result/$work_dir.$name.$opt.$lr
# done
# done

# tensorboard dev upload --logdir experiments/$work_dir-10