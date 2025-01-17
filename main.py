import argparse
import os
import time
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

from models import *
from utils.plot_utils import get_logger, calc_grad_norm, AverageMeter


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

parser.add_argument("--name", type=str, default="N/A",
                        help="experiment name")
parser.add_argument("--work_dir", default="ResNetexps", type=str,
                    help="experiment directory.")
parser.add_argument("--save_dir", default="result/preact_resnet110_cifar", type=str,
                    help="where to save model") 
parser.add_argument("--model_type", choices=[
                        "wtii_deq_preact_resnet110_cifar", 
                        "deq_parresnet110_cifar",
                        "preact_resnet110_cifar",
                        "wtii_preact_resnet110_cifar"],
                    help="type of the model (sequential, parallel)")
parser.add_argument('--resume', default='', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')   
parser.add_argument('--load_optim', action="store_true",
                    help='should optimizer state be loaded')           
         

parser.add_argument("--seed", type=int, default=228,
                    help="random seed")
parser.add_argument('--epochs', default=200, type=int, metavar='N', 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', 
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', 
                    help='initial learning rate')
parser.add_argument('--optimizer', default="sgd", choices=["adam", "sgd"],
                    help='optimizer type')                
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (sgd)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', 
                    help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', 
                    help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', 
                    help='10 for cifar10,100 for cifar100 (default: 10)')

parser.add_argument('--wnorm', '--wn', action="store_true", 
                    help='weight normalization (do not use mess it with weight decay)')
parser.add_argument('--norm_func', default='batch', choices=get_norm_func().keys(), 
                    help='normalization function for resnet block')
parser.add_argument("--identity_mapping", action="store_true", 
                        help="residual path is clear as in PreResNet") 
parser.add_argument("--inplanes", type=int, default=16,
                        help="width of the model")
parser.add_argument("--dropout", type=float, default=0.0,
                        help="Variational dropout rate")
parser.add_argument("--track_running_stats", action="store_true",
                        help="Variational dropout rate")
parser.add_argument("--layers", type=int, default=18,
                        help="layers (aka blocks) of WT model")          

parser.add_argument('--n_layer', type=int, default=18,
                    help='number of total layers')
parser.add_argument('--f_thres', type=int, default=30,
                    help='forward pass Broyden threshold')
parser.add_argument('--b_thres', type=int, default=10000,
                    help='backward pass Broyden threshold')
parser.add_argument('--pretrain_steps', type=int, default=200,
                    help='number of pretrain steps')
parser.add_argument('--clip', type=float, default=10.0,
                    help='gradient clipping (default: None)')



parser.add_argument('--debug',  action='store_true')
parser.add_argument('--max_train_it',  default=1, type=int)
parser.add_argument('--max_test_it',  default=1, type=int)

parser.add_argument('--test_mode', default="broyden", choices=["broyden", "forward"],
                    help="mode for test/validation (actually should be just 'mode')")
parser.add_argument('--store_trajs', action='store_true',
                    help="if store forward trajectories of broyden")

parser.add_argument('--midplanes', default=16, type=int)
parser.add_argument('--skip_block', action='store_true',
                    help='if use only one block')

                          

best_prec = 0
train_global_it = 0
test_global_it = 0

def main():
    global args, best_prec, logging, writer
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        print("running in cpu mode!")
    use_gpu = torch.cuda.is_available()

    args.name += f"_normf{args.norm_func}" \
              +  f"_inpls{args.inplanes}" \
              +  f"_midpls{args.midplanes}" \
              +  f"_trs{args.track_running_stats}" \
              +  f"_wnm{args.wnorm}" \
              +  f"_prets{args.pretrain_steps}" \
              +  f"_nlayer{args.n_layer}" \
              +  f"_fth{args.f_thres}" \
              +  f"_optim{args.optimizer}" \
              +  f"_lr{args.lr}"
    print(f"Experiment name: {args.name}")
    args.work_dir = '{}-{}'.format(args.work_dir, args.cifar_type)
    args.work_dir = os.path.join(args.work_dir, "{}-{}".format(args.name, time.strftime('%Y-%m-%d--%H-%M-%S')))
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    print('Experiment dir : {}'.format(args.work_dir))
    logging = get_logger(os.path.join(args.work_dir, "log.txt"), print_=args.debug)
    writer = SummaryWriter(args.work_dir, flush_secs=1)


    # Model building
    logging('=> Building model...')
    if use_gpu:
        import multiprocessing as mp 
        mp.set_start_method('spawn')
        # https://github.com/pytorch/pytorch/issues/2517
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(args.seed)

        # model can be set to anyone that I have defined in models folder
        # note the model should match to the cifar type !


        # wtii_deq_preact_resnet110_cifar
        #deq_parresnet110_cifar
        model = eval(args.model_type)(wnorm=args.wnorm, 
            pretrain_steps=args.pretrain_steps,
            inplanes=args.inplanes,
            midplanes=args.midplanes,
            norm_func=args.norm_func,
            track_running_stats=args.track_running_stats,
            n_layer=args.n_layer,
            test_mode=args.test_mode,
            skip_block=args.skip_block,
        )
        # model = preact_resnet110_cifar(num_classes=10, inplanes=16)

        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = args.save_dir
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # adjust the lr according to the model type
        if isinstance(model, (ResNet_Cifar, PreAct_ResNet_Cifar, WTIIPreAct_ResNet_Cifar, WTIIPreAct_ParResNet_Cifar, DEQParResNet)):
            model_type = 1
        else:
            logging('model type unrecognized...')
            return

        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        cudnn.benchmark = True
    else:
        logging('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            logging('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            if args.load_optim:
                optimizer.load_state_dict(checkpoint['optimizer'])
            logging("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logging("=> no checkpoint found at '{}'".format(args.resume))

    args.n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])

    logging('=' * 100)
    for k, v in args.__dict__.items():
        logging('    - {} : {}'.format(k, v))
    logging('=' * 100)
    logging(f'#params = {args.n_all_param}')

    logging(str(model))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        logging('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # CIFAR100
    else:
        logging('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        if args.store_trajs:
            with open(os.path.join(args.work_dir, "trajs.pkz"), "wb") as f:
                validate(testloader, model, criterion, store_trajs=f)
        else:
            validate(testloader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, model_type)

        # train for one epoch
        train(trainloader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec = validate(testloader, model, criterion)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)

        writer.add_scalar('precison_by_epoch', 
                                prec,
                                epoch)
    writer.close()



def train(trainloader, model, criterion, optimizer, epoch):
    global train_global_it
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()
        # compute output
        output = model(input, train_step=train_global_it, f_thres=args.f_thres,
                                        b_thres=args.b_thres)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if isinstance(model.module, (WTIIPreAct_ResNet_Cifar, WTIIPreAct_ParResNet_Cifar)):
            model.module.update_meters(input)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0 and isinstance(model.module, (WTIIPreAct_ResNet_Cifar, WTIIPreAct_ParResNet_Cifar)):
            _, debug_info = model.module(input, debug=True)
            if debug_info is not None: # if batch size is correct
                logging("DEBUG TRAIN:" + debug_info)

        if i % args.print_freq == 0:
            logging('Epoch: [{0}][{1}/{2}] \tGlobal[{it}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%) \t'.format(
                   epoch, i, len(trainloader), it=train_global_it, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            writer.add_scalar('train/loss', 
                                losses.val,
                                train_global_it)
            writer.add_scalar('train/top1', 
                                top1.val,
                                train_global_it)
            if isinstance(model.module, (WTIIPreAct_ResNet_Cifar, WTIIPreAct_ParResNet_Cifar)):
                diffs = model.module.get_diffs()
                if diffs is not None:
                    for mode in diffs:
                        diff = diffs[mode]
                        for layer in diff:
                            meter = diff[layer]
                            if meter.val is not None:
                                writer.add_scalar(f'train/{mode}_{layer}',
                                            meter.val, 
                                            train_global_it)  
                grads = model.module.get_grads()
                if grads is not None:
                    for key in grads:
                        if grads[key].val is not None:
                            writer.add_scalar(f'train/grad{key}',
                                        grads[key].val, 
                                        train_global_it)                 
            train_global_it += 1
            if args.debug and train_global_it >= args.max_train_it:
                break


def validate(val_loader, model, criterion, store_trajs=None):
    global test_global_it
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input, f_thres=args.f_thres,
                                        b_thres=args.b_thres, store_trajs=store_trajs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))
            if isinstance(model.module, (WTIIPreAct_ResNet_Cifar, WTIIPreAct_ParResNet_Cifar)):
                model.module.update_meters(input)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if store_trajs is None and i == 0 and isinstance(model.module, (WTIIPreAct_ResNet_Cifar, WTIIPreAct_ParResNet_Cifar)):
                _, debug_info = model.module(input[:1, ...], debug=True)
                if debug_info is not None: # if batch size is correct
                    logging("DEBUG TEST:" + debug_info)

            if i % args.print_freq == 0:
                logging('Test: [{0}/{1}] \tGlobal[{it}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)\t'.format(
                   i, len(val_loader), it=test_global_it, batch_time=batch_time, loss=losses,
                   top1=top1))
                writer.add_scalar('test/loss', 
                                    losses.val,
                                    test_global_it)
                writer.add_scalar('test/top1',
                                    top1.val,
                                    test_global_it)
                if isinstance(model.module, (WTIIPreAct_ResNet_Cifar, WTIIPreAct_ParResNet_Cifar)):
                    diffs = model.module.get_diffs()
                    if diffs is not None:
                        for mode in diffs:
                            diff = diffs[mode]
                            for layer in diff:
                                meter = diff[layer]
                                if meter.val is not None:
                                    writer.add_scalar(f'test/{mode}_{layer}',
                                                meter.val, 
                                                test_global_it) 
                test_global_it += 1
                if args.debug and test_global_it >= args.max_test_it:
                    break
            if store_trajs is not None:
                return

    logging(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, model_type):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if model_type == 1:
        if epoch < 80:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif model_type == 2:
        if epoch < 60:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.2
        elif epoch < 160:
            lr = args.lr * 0.04
        else:
            lr = args.lr * 0.008
    elif model_type == 3:
        if epoch < 150:
            lr = args.lr
        elif epoch < 225:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    main()

