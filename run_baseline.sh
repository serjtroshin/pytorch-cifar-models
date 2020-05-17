#!/bin/bash

# forward

work_dir=baseline-PreActResNet110


name=$work_dir
sbatch -c 3 -G 1 run.sh --epoch 160 --batch-size 128 -ct 10 \
--name $name \
--optimizer sgd --lr 0.1 \
--inplanes 16 \
--midplanes 16 \
--work_dir experiments/$work_dir \
--track_running_stats \
--model_type preact_resnet110_cifar \
--save_dir result/$name

 

tensorboard dev upload --logdir experiments/$work_dir-10