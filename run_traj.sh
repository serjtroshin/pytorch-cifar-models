# parallel
# python main.py --resume \
# deq_models/result/DEQparallel-low_n_layer_pretraining.exp..0.03.adam.0.03.61/checkpoint.pth \
# --work_dir notebooks/trajectories \
# --name DEQparallel_traj_forward \
# --f_thres 30 \
# --inplanes 61 \
# --track_running_stats \
# --evaluate \
# --lr 0.03 \
# --optim adam \
# --test_mode forward \
# --pretrain_steps 1000000 \
# --model_type deq_parresnet110_cifar \
# --store_traj


python main.py --resume \
deq_models/result/DEQsequential-3blocks.sequential_broyden.adam.0.001.61/checkpoint.pth  \
--work_dir notebooks/trajectories \
--name DEQseqential_traj_forward \
--f_thres 30 \
--inplanes 61 \
--track_running_stats \
--evaluate \
--lr 0.001 \
--optim adam \
--test_mode forward \
--model_type wtii_deq_preact_resnet110_cifar \
--store_traj