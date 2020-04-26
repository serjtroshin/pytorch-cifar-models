#!/bin/bash

# forward

work_dir=DEQsequential-plot-trajectory

# for optim in sgd adam
optim=adam
lr=0.001
sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 -ct 10 \
--name test_broyden \
--optimizer $optim \
--lr $lr \
--pretrain_steps 0 \
--test_mode broyden \
--resume pretrained_models/adam.0.001.oneblock/checkpoint.pth \
--work_dir experiments/$work_dir \
--evaluate \
--f_thres 1000 \
--store_trajs


# tensorboard dev upload --logdir experiments/$work_dir-10