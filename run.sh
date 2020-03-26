set -e

cat $0

#python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
#--name baseline

#python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
#--name wnorm --wnorm --weight-decay 0.0

#python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
#--name identity_mapping --identity_mapping

#python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
#--name inst --norm_func inst

#python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
#--name inst_identity_mapping --norm_func inst --identity_mapping

#CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
#--name inst_identity_mapping_wnorm --norm_func inst --identity_mapping --wnorm &

#CUDA_VISIBLE_DEVICES=1 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
#--name inst_wnorm --norm_func inst --wnorm &

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
--name inst_identity_mapping_wnorm_inplanes42 --norm_func inst --identity_mapping --wnorm --inplanes 42 &

CUDA_VISIBLE_DEVICES=1 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
--name inst_wnorm_inplanes42 --norm_func inst --wnorm --inplanes 42 &

CUDA_VISIBLE_DEVICES=2 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
--name inst_inplanes42 --norm_func inst --inplanes 42 &

CUDA_VISIBLE_DEVICES=3 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10 \
--name inst_identity_mapping_inplanes42 --norm_func inst --identity_mapping --inplanes 42 &


