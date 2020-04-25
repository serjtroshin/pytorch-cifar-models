#!/bin/bash
rm -r experiments/tmp-10

sbatch -c 4 -G 1 run.sh --epoch 10 --batch-size 128 --optimizer sgd -ct 10 \
--name test --debug --test_mode broyden --pretrain_steps 0 --work_dir experiments/tmp \
--resume pretrained_models/checkpoint.pth

tensorboard dev upload --logdir experiments/tmp-10
