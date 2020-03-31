set -e

cat $0

CUDA_VISIBLE_DEVICES=2 python main.py --epoch 160 --batch-size 128 --lr 0.01 --momentum 0.9 -ct 10 \
--name baseline_par_fixed_batchnorm_lr0.01 --work_dir ResNetParexps-10