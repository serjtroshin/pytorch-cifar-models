#!/bin/sh
#set -e

cat $0

python main.py --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
--name baseline_par_fixed_batchnorm_additive_track_running_statsFalse_layers18 --work_dir ResNetParexps