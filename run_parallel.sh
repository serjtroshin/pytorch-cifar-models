#!/bin/bash


sbatch -c 4 -G 2 run.sh --epoch 160 --batch-size 256 --optimizer sgd -ct 10 \
--norm_func inst --f_thres 32 \
--name deqinst_clipgrad_and_adam --pretrain_steps 200 --work_dir experiments/DEQparallel

sbatch -c 16 -G 4 run.sh --epoch 160 --batch-size 256 --optimizer sgd -ct 10 \
--norm_func inst --f_thres 64 \
--name deqinst_clipgrad_and_adam --pretrain_steps 200 --work_dir experiments/DEQparallel