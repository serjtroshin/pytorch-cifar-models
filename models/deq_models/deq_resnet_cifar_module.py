import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np

from models.deq_models.deq_modules.deq import *


class ParResNetDEQModule(DEQModule):

    """ See DEQModule class for documentation """

    def __init__(self, func, func_copy):
        super(ParResNetDEQModule, self).__init__(func, func_copy)
        
    def forward(self, z1s, us, z0, **kwargs):
        bsz, total_hsize, seq_len = z1s.size()
        train_step = kwargs.get('train_step', -1)
        threshold = kwargs.get('threshold', None)
        debug = kwargs.get('debug', False)
        store_trajs = kwargs.get('store_trajs', None)

        if us is None:
            raise ValueError("Input injection is required.")
        # assert threshold == 50, threshold
        z1s_out = RootFind.apply(self.func, z1s, us, z0, store_trajs, debug, threshold, train_step)
        if self.training:
            z1s_out = RootFind.f(self.func, z1s_out, us, z0, threshold, train_step)
            self.Backward._need_grad_wrt_us = True
            z1s_out = self.Backward.apply(self.func_copy, z1s_out, us, z0, threshold, train_step)
        return z1s_out