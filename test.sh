#!/bin/bash
rm -r experiments/tmp-10

bash run.sh --epoch 20 --batch-size 128 --optimizer sgd -ct 10 \
--name test --pretrain_steps 10 --work_dir experiments/tmp

