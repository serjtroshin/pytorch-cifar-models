#!/bin/bash
rm -r experiments/tmp-10

bash run.sh --epoch 2 --batch-size 128 --optimizer sgd -ct 10 \
--name test --debug --test_mode broyden --pretrain_steps 1 --work_dir experiments/tmp

