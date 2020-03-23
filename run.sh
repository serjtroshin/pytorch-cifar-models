set -e

cat $0

python main.py \
--epoch 160 \
--batch-size 128 \
--lr 0.1 \
--momentum 0.9 \
-ct 10 \
--name baseline
