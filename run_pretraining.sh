#!/bin/bash

# forward

work_dir=DEQsequential-pretraining


sbatch -c 4 -G 1  run.sh --epoch 10 --batch-size 128 -ct 10 \
--name train_pretrain_model \
--optimizer sgd --lr 0.1 \
--test_mode forward \
--n_layer 5 \
--work_dir experiments/$work_dir \
--save_dir pretrained_models/pretrained_5layer_oneblock

tensorboard dev upload --logdir experiments/$work_dir-10