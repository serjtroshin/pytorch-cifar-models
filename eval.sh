# parallel
for thr in 1 2 3 4 5 6 7 8 9 10
do
python main.py --resume \
deq_models/result/DEQparallel-low_n_layer_pretraining.exp..0.03.adam.0.03.61/checkpoint.pth \
--name DEQparallel-forward \
--n_layer $thr \
--inplanes 61 \
--track_running_stats \
--test_mode forward \
--model_type deq_parresnet110_cifar \
--evaluate
done

#sequential
for thr in 1 2 3 5 7 9 12 15 20 30 40 50 100
do
python main.py --resume \
deq_models/result/DEQsequential-3blocks.sequential_broyden.adam.0.001.61/checkpoint.pth \
--name DEQsequential-forward \
--n_layer $thr \
--inplanes 61 \
--track_running_stats \
--test_mode forward \
--model_type wtii_deq_preact_resnet110_cifar \
--evaluate
done