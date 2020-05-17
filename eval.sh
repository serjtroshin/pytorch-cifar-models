work_dir=eval

# parallel
# for thr in 1 2 3 4 5 6 7 8 9 10
# do
# python main.py --resume \
# deq_models/result/DEQparallel-low_n_layer_pretraining.exp..0.03.adam.0.03.61/checkpoint.pth \
# --name DEQparallel-forward \
# --n_layer $thr \
# --inplanes 61 \
# --track_running_stats \
# --test_mode forward \
# --model_type deq_parresnet110_cifar \
# --evaluate
# done

#sequential
for thr in 60 100
do
for mode in broyden forward
do
sbatch -c 3 -G 1 run.sh --resume result/DEQsequential-continue.sequential_broyden.adam.0.001/checkpoint.pth \
--name DEQsequential-$mode \
--n_layer $thr \
--f_thres $thr \
--optim adam --lr 0.001 \
--test_mode $mode \
--inplanes 61 \
--midplanes 61 \
--track_running_stats \
--model_type wtii_deq_preact_resnet110_cifar \
--work_dir experiments/$work_dir \
--evaluate
done
done

# tensorboard dev upload --logdir experiments/$work_dir-10