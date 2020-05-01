#!/bin/bash
# rm -r experiments/tmp-10

sbatch -c 4 -G 1 run.sh --epoch 1 --batch-size 1000 --optimizer sgd -ct 10 \
--name profile --test_mode broyden --pretrain_steps 0 --inplanes 42 --work_dir experiments/tmp


# tensorboard dev upload --logdir experiments/tmp-10
