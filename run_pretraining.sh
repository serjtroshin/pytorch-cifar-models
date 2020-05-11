#!/bin/bash

# forward

work_dir=DEQsequential-pretraining-grid-final

for opt in adam
do
for lr in 0.1 0.01 0.001 0.0001
do
sbatch -c 3 -G 1  run.sh --epoch 10 --batch-size 128 -ct 10 \
--name train_pretrain_model \
--optimizer $opt --lr $lr \
--test_mode forward \
--n_layer 5 \
--inplanes 64 \
--midplanes 64 \
--work_dir experiments/$work_dir \
--track_running_stats \
--model_type wtii_deq_preact_resnet110_cifar \
--save_dir pretrained_models/pretrained_5layer_oneblock64.$opt.$lr.track_running_stats
done
done

tensorboard dev upload --logdir experiments/$work_dir-10