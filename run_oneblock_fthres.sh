#!/bin/bash

# forward

work_dir=DEQ-oneblock-fthres

name=$work_dir
for opt in adam
do
for lr in 0.1
do
for fthr in 1 3 5 12 30 60
do
sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 10 \
--name $name.lr.$lr \
--optimizer $opt --lr $lr \
--test_mode broyden \
--f_thres $fthr \
--n_layer 5 \
--inplanes 64 \
--midplanes 64 \
--skip_block \
--pretrain_steps 0 \
--work_dir experiments/$work_dir \
--track_running_stats \
--resume pretrained_models/pretrained_5layer_oneblock64.$opt.0.1.track_running_stats/checkpoint.pth \
--model_type wtii_deq_preact_resnet110_cifar \
--save_dir result/$work_dir.$name.$opt.$lr.64
done
done
done
 

# tensorboard dev upload --logdir experiments/$work_dir-10