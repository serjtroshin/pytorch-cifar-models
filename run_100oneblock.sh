#!/bin/bash

# forward

work_dir=baseline-oneblock


name=$work_dir
for opt in sgd
do
for lr in 0.1
do
for inplanes in 21
do
bash run.sh --epoch 160 --batch-size 128 -ct 100 \
--name $name \
--optimizer $opt --lr $lr \
--test_mode broyden \
--inplanes $inplanes \
--midplanes $inplanes \
--work_dir experiments/$work_dir \
--track_running_stats \
--skip_block \
--model_type preact_resnet110_cifar \
--save_dir result/$work_dir.$name.$opt.$lr.inplanes.$inplanes
done
done
done

# name=$work_dir.wtii
# for opt in sgd
# do
# for lr in 0.1
# do
# sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 10 \
# --name $name \
# --optimizer $opt --lr $lr \
# --test_mode forward \
# --n_layer 18 \
# --inplanes 64 \
# --midplanes 64 \
# --skip_block \
# --work_dir experiments/$work_dir \
# --track_running_stats \
# --model_type wtii_preact_resnet110_cifar \
# --skip_block \
# --save_dir result/$work_dir.$name.$opt.$lr.wtii.inplanes.64
# done
# done

# name=$work_dir
# for opt in adam
# do
# for lr in 0.01 0.001 0.1
# do
# for prelr in 0.001 0.1 0.01 
# do
# sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 10 \
# --name $name.lr.$lr.prelr.$prelr \
# --optimizer $opt --lr $lr \
# --test_mode broyden \
# --n_layer 5 \
# --inplanes 64 \
# --midplanes 64 \
# --skip_block \
# --pretrain_steps 0 \
# --work_dir experiments/$work_dir \
# --track_running_stats \
# --model_type wtii_deq_preact_resnet110_cifar \
# --resume pretrained_models/pretrained_5layer_oneblock64.$opt.$prelr.track_running_stats/checkpoint.pth \
# --save_dir result/$work_dir.$name.$opt.$lr.$prelr
# done
# done
# done

# tensorboard dev upload --logdir experiments/$work_dir-10