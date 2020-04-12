#!/bin/bash


# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func batch --inplanes 47 --name parallel --work_dir ResNetParexpsall

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func batch --inplanes 16 --name parallel --work_dir ResNetParexpsall

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func inst --inplanes 47 --name parallel --work_dir ResNetParexpsall

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func inst --inplanes 16 --name parallel --work_dir ResNetParexpsall

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func batch --inplanes 47 --track_running_stats --name parallel --work_dir ResNetParexpsall

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func batch --inplanes 16 --track_running_stats --name parallel --work_dir ResNetParexpsall

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func inst --inplanes 47 --track_running_stats --name parallel --work_dir ResNetParexpsall

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func inst --inplanes 16 --track_running_stats --name parallel --work_dir ResNetParexpsall

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func batch --inplanes 47 --wnorm --wd 0.0 --name parallel --work_dir ResNetParexpsall

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func inst --inplanes 47 --wnorm --wd 0.0 --name parallel --work_dir ResNetParexpsall

# for layer in 3 7 10 18 30 50
# do
# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
# --norm_func inst --inplanes 47 --layers $layer --name parallel --work_dir ResNetParexp_layers
# done

# sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --optimizer adam -ct 10 \
# --norm_func inst --inplanes 47 --name parallel --work_dir ResNetParexpall

sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --optimizer sgd -ct 10 \
--name deqparallel --work_dir experiments/DEQparallel