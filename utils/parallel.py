import torch
from torch import nn
from .plot_utils import ConvergenceMeter
import json


class Parallel(nn.Module):
    def __init__(self, transitions):
        super(Parallel, self).__init__()
        self._n = len(transitions)
        self._m = len(transitions[0])
        for i in range(len(transitions)):
            for j in range(len(transitions[0])):
                setattr(self, f'transition_f_{i}_{j}', transitions[i][j])
        self.meters = {}
        self.info = {
            "pretrain_diffs" : None,
            "forward_diffs" : None,
            "backward_diffs" : None,
            "forward_result_info" : None,
            "backward_result_info" : None
        }
        self.meter = ConvergenceMeter()

    def reset(self):
        for meter in self.meters.values():
            meter.reset() # do not forget to reset

    def copy(self, func):
        self.info = func.info  # shared
        for i in range(self._n):
            for j in range(self._m):
                elem1 = getattr(self, f'transition_f_{i}_{j}')
                elem2 = getattr(func, f'transition_f_{i}_{j}')
                if elem1 is not None:
                    elem1.copy(elem2)   

    def get_diffs(self):
        res = self.info.copy()
        self.info = {
            "pretrain_diffs" : None,
            "forward_diffs" : None,
            "backward_diffs" : None,
            "forward_result_info" : None,
            "backward_result_info" : None
        }
        if len(self.meter.diffs) != 0:
            info = {"pretrain_info" : str(list(map(lambda x : f"{x:.4f}", self.meter.diffs)))}
            res.update(info)
            self.meter.reset()
        return json.dumps(res, sort_keys=True, indent=4)


    def forward(self, x, zs, channels_dim=1, debug=False):
        zs_next = []
        for i in range(len(zs)):
            z_input = None
            z_other = []
            for j, zj in enumerate(zs):
                if j == i: 
                    z_input = zj
                    continue
                if getattr(self, f'transition_f_{j}_{i}') is None:
                    continue
                z_other.append(getattr(self, f'transition_f_{j}_{i}')(zj))
            if i == 0:
                z_other = sum(z_other + [x])
            else:
                z_other = sum(z_other)
            z_next = getattr(self, f'transition_f_{i}_{i}')(z_input + z_other)
            if debug:
                if i not in self.meters:
                    self.meters[i] = ConvergenceMeter()
                self.meters[i].update(z_next)
            zs_next.append(z_next)
        return x, zs_next

