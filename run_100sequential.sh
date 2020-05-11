#!/bin/bash

# forward

work_dir=DEQsequential-baseline

# name=forward
# for opt in sgd
# do
# for lr in 0.1
# do
# for inplanes in 16
# do
# sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 100 \
# --name $name \
# --optimizer $opt --lr $lr \
# --n_layer 18 \
# --inplanes 16 \
# --work_dir experiments/$work_dir \
# --track_running_stats \
# --model_type preact_resnet164_cifar \
# --save_dir result/$work_dir.$name.$opt.$lr.$inplanes
# done
# done
# done


name=wtii
for opt in sgd
do
for lr in 0.1
do
for inplanes in 16
do
sbatch -c 3 -G 1 -t 6320 run.sh --epoch 160 --batch-size 128 -ct 100 \
--name $name \
--optimizer $opt --lr $lr \
--n_layer 18 \
--inplanes 16 \
--midplanes 84 \
--work_dir experiments/$work_dir \
--track_running_stats \
--model_type wtii_preact_resnet164_cifar \
--save_dir result/$work_dir.$name.$opt.$lr.$inplanes
done
done
done

# tensorboard dev upload --logdir experiments/$work_dir-100