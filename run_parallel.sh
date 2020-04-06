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

sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
--norm_func batch --inplanes 47 --wnorm --wd 0.0 --name parallel --work_dir ResNetParexpsall

sbatch -c 4 -G 1 run.sh --epoch 160 --batch-size 128 --momentum 0.9 -ct 10 \
--norm_func inst --inplanes 47 --wnorm --wd 0.0 --name parallel --work_dir ResNetParexpsall