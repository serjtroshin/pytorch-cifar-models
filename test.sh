#!/bin/bash
rm -r experiments/tmp-10

sbatch -c 4 -G 1 run.sh --epoch 2 --batch-size 128 --optimizer sgd -ct 10 \
--name test --pretrain_steps 1 --work_dir experiments/tmp

