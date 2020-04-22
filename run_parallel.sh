#!/bin/bash

# forward

sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer sgd -ct 10 \
--name test_seq_no_clipping_sgd_FORWARD --f_thres 30 --pretrain_steps 3000000 \
--clip 1000 --testmode forward --work_dir experiments/DEQsequential-fullmodel

# broyden

sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer sgd -ct 10 \
--name test_seq_sgd_BROYDEN --f_thres 50 --pretrain_steps 300 --clip 50 --work_dir experiments/DEQsequential-fullmodel

# shallow pretraining

sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer sgd -ct 10 \
--name test_seq_sgd_shallow_BROYDEN --f_thres 50 --pretrain_steps 300 --n_layer 5 --clip 50 --work_dir experiments/DEQsequential-fullmodel

# broyden adam

sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
--name test_seq_sgd_BROYDEN --f_thres 50 --pretrain_steps 300 --clip 50 --work_dir experiments/DEQsequential-fullmodel

# shallow pretraining adam

sbatch -c 4 -G 1  run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
--name test_seq_sgd_shallow_BROYDEN --f_thres 50 --pretrain_steps 300 --n_layer 5 --clip 50 --work_dir experiments/DEQsequential-fullmodel