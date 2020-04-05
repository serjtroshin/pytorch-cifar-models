#!/bin/sh
#set -e

cat $0

python main.py --epoch 3 --batch-size 128 --momentum 0.9 -ct 10 \
--name baseline_par_fixed_batchnorm_additive --work_dir ResNetParexps-10