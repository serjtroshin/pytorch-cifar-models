# Experiments on CIFAR datasets with PyTorch with DEQ

## Introduction
Reimplement state-of-the-art CNN models in cifar dataset with PyTorch, now including:
[PreActResNet](https://arxiv.org/abs/1603.05027v3)


## Requirements:software
Requirements for [PyTorch](http://pytorch.org/)

## Requirements:hardware
For most experiments, one or two K40(~11G of memory) gpus is enough cause PyTorch is very memory efficient. However,
to train DenseNet on cifar(10 or 100), you need at least 4 K40 gpus.

## Usage
1. Clone this repository

```
git clone https://github.com/serjtroshin/pytorch-cifar-models.git
```


2. Edit main.py

In the ```main.py```, you can specify the network you want to train(for example):

```
model = resnet20_cifar(num_classes=10)

Then, you need specify some parameter for training in run.sh script

```
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct 10
```

3. Train

```
nohup sh run.sh > resnet20_cifar10.log &
```

After training, the training log will be recorded in the .log file, the best model(on the test set) 
will be stored in the fdir.

**Note**:For first training, cifar10 or cifar100 dataset will be downloaded, so make sure your comuter is online.
Otherwise, download the datasets and decompress them and put them in the ```data``` folder.

4. Test

```
CUDA_VISIBLE_DEVICES=0 python main.py -e --resume=fdir/model_best.pth.tar
```

5. CIFAR100

The default setting in the code is for cifar10, to train with cifar100, you need specify it explicitly in the code.

```
model = resnet20_cifar(num_classes=100)
```

**Note**: you should also change **fdir** In the run.sh, you should set ```-ct 100```




# References:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[3] Bai S.,Kolter J. Z.,Koltun V.Deep Equilibrium Models. — 2019. —arXiv:1909.01377 [cs.LG]. In NeurIPS 2019.
