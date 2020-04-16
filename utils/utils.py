from torch import nn
from .plot_utils import ConvergenceMeter

class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)
        self.meter = ConvergenceMeter()

    def _reset(self, *input):
        self.meter.reset()

    def wnorm(self):
        list(self._modules.values())[1].wnorm() # set wnorm only once

    def forward(self, *input, debug=False):
        self._reset(*input)
        list(self._modules.values())[0].dropout.reset_mask(input[1])
        input = list(self._modules.values())[0](*input)
        list(self._modules.values())[1].dropout.reset_mask(input[1])
        list(self._modules.values())[1].reset() # reset layer only once
        for module in list(self._modules.values())[1:]:
            input = module(*input)
            if debug:
                self.meter.update(input[0])
        if debug:
            return input, self.meter.diffs
        return input, None


class SequentialLayer(nn.Module):
    def __init__(self, layer):
        super(SequentialLayer, self).__init__()
        self.meter = ConvergenceMeter()
        self.layer = layer

    def reset(self):
        self.meter.reset()

    def wnorm(self):
        self.layer.wnorm()

    def copy(self, func):
        self.layer.copy(func.layer)

    def get_diffs(self):
        return self.meter.diffs

    def forward(self, *input, debug=False):
        input = self.layer(*input)
        if debug:
            self.meter.update(input[0])
        return input