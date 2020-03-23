cat $0
CUDA_VISIBLE_DEVICES=1,2,3 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 -ct 10
