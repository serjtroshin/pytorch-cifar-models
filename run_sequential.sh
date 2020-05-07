#!/bin/bash

# forward

work_dir=DEQsequential-3midplanes

# name=sequential_broyden
# for opt in adam
# do
# for lr in 0.1 0.01 0.001
# do
# sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 10 \
# --name $name \
# --optimizer $opt --lr $lr \
# --test_mode broyden \
# --n_layer 5 \
# --inplanes 61 \
# --pretrain_steps 20 \
# --work_dir experiments/$work_dir \
# --track_running_stats \
# --save_dir result/$work_dir.$name.$opt.$lr.61
# done
# done


name=sequential_forward
for opt in sgd
do
for lr in 0.1
do
bash run.sh --epoch 160 --batch-size 64 -ct 10 \
--name $name \
--optimizer $opt --lr $lr \
--test_mode forward \
--n_layer 18 \
--inplanes 16 \
--midplanes 100 \
--pretrain_steps 200000 \
--work_dir experiments/$work_dir \
--track_running_stats \
--save_dir result/$work_dir.$name.$opt.$lr.inplanes16.midplanes100
done
done

# tensorboard dev upload --logdir experiments/$work_dir-10