#!/bin/bash


bash run.sh --epoch 160 --batch-size 128 --optimizer sgd -ct 10 \
--name test_seq --pretrain_steps 50000 --work_dir experiments/DEQsequential-test