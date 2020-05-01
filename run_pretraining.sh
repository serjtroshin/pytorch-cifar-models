#!/bin/bash

# forward

work_dir=DEQsequential-pretraining-grid

for opt in adam sgd
do
for lr in 0.1 0.01 0.001
do
sbatch -c 4 -G 1  run.sh --epoch 10 --batch-size 128 -ct 10 \
--name train_pretrain_model \
--optimizer $opt --lr $lr \
--test_mode forward \
--n_layer 5 \
--inplanes 64 \
--work_dir experiments/$work_dir \
--save_dir pretrained_models/pretrained_5layer_oneblock64.$opt.$lr
done
done

tensorboard dev upload --logdir experiments/$work_dir-10