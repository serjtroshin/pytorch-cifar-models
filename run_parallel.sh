#!/bin/bash


bash run.sh --epoch 160 --batch-size 128 --optimizer sgd -ct 10 \
--name test_seq --pretrain_steps 50 --optimizer adam --lr 1e-3 --work_dir experiments/DEQsequential-test