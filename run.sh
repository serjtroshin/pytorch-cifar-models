set -e

cat $0

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
--name inst_identity_mapping_wnorm_inplanes42_indropout0.1 --norm_func inst --identity_mapping --wnorm --inplanes 42 \
--dropout 0.1 &

CUDA_VISIBLE_DEVICES=1 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
--name inst_wnorm_inplanes42_indropout0.1 --norm_func inst --wnorm --inplanes 42 \
--dropout 0.1 &
