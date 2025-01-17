'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math
import sys
from functools import partial

sys.path.append('../')
from utils.parallel import Parallel
from utils.optimization import weight_norm, VariationalDropout


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def deconv3x3(in_planes, out_planes, stride=1):
    " 3x3 deconvolution with padding "
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)

def get_norm_func():
    return {
        "inst" : partial(nn.InstanceNorm2d, affine=False),
        "batch" : partial(nn.BatchNorm2d, affine=False)
    }

class PreActBasicParBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, midplanes, planes, stride=1, norm_func=nn.BatchNorm2d, identity_mapping=False):
        super(PreActBasicParBlock, self).__init__()
        self.bn1 = norm_func(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn2 = norm_func(midplanes)
        self.conv2 = conv3x3(midplanes, planes)
        self.stride = stride

        self.identity_mapping = identity_mapping

        if not self.identity_mapping:
            self.norml = norm_func(planes)

    def wnorm(self):
        self.conv1, self.conv1_fn = weight_norm(module=self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(module=self.conv2, names=['weight'], dim=0)

    def copy(self, func):
        self.conv1.weight.data = func.conv1.weight.data.clone()
        self.conv2.weight.data = func.conv2.weight.data.clone()

    def reset(self):
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)

    def forward(self, z):
        residual = z
        
        out = self.bn1(z)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        if not self.identity_mapping:
            out = self.norml(out)


        return out


class TransitionBlock(nn.Module):
    expansion = 1

    def wnorm(self):
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Conv2d):
                conv, conv_fn = weight_norm(module=module, names=['weight'], dim=0)
                self.net[i] = conv
                self.net[i]._fn = conv_fn

    def reset(self):
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Conv2d):
                if '_fn' in self.net[i].__dict__:
                   module._fn.reset(module)

    def copy(self, func):
        self.net[2].weight.data = func.net[2].weight.data.clone()

    def __init__(self, inplanes, planes, norm_func=nn.BatchNorm2d, layers=2, transition_f=conv3x3):
        super(TransitionBlock, self).__init__()
        assert layers == 1, "todo make copy for layers = 2"
        inter_planes = (inplanes + planes) // 2
        inter_layers = layers - 2
        net = []
        if layers == 1:
            net.extend([
                norm_func(inplanes),
                nn.ReLU(inplace=True),
                transition_f(inplanes, planes, stride=2)
            ])
        else:
            net.extend([
                norm_func(inplanes),
                nn.ReLU(inplace=True),
                transition_f(inplanes, inter_planes, stride=2)
            ])
            for i in range(inter_layers):
                net.extend([
                    norm_func(inter_planes),
                    nn.ReLU(inplace=True),
                    transition_f(inter_planes, inter_planes, stride=2)
                ])
            net.extend([
                norm_func(inter_planes),
                nn.ReLU(inplace=True),
                transition_f(inter_planes, planes, stride=2)
            ])
        self.net = nn.Sequential(*net)

    def forward(self, z):
        return self.net(z)

class UpBlock(TransitionBlock):
    expansion = 1
    def __init__(self, inplanes, planes, norm_func=nn.BatchNorm2d, layers=2):
        super(UpBlock, self).__init__(inplanes, planes, norm_func, layers, 
                                        transition_f=deconv3x3)

class DownBlock(TransitionBlock):
    expansion = 1
    def __init__(self, inplanes, planes, norm_func=nn.BatchNorm2d, layers=2):
        super(DownBlock, self).__init__(inplanes, planes, norm_func, layers, 
                                        transition_f=conv3x3)


class WTIIPreAct_ParResNet_Cifar(nn.Module):

    def __init__(self, block, down_block, up_block, layers, num_classes=10, **kwargs):
        super(WTIIPreAct_ParResNet_Cifar, self).__init__()

        track_running_stats=kwargs.get("track_running_stats", False)
        self.norm_func=partial(get_norm_func()[kwargs.get("norm_func", "batch")], track_running_stats=track_running_stats)
        wnorm=kwargs.get("wnorm", False) # weight normalization
        # self.identity_mapping=kwargs.get("identity_mapping", False) # is identity path clear
        self.inplanes=kwargs.get("inplanes", 16)
        midplanes=kwargs.get("midplanes", 16)

        inplanes = self.inplanes

        self.layers = layers

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.transitions = [[None] * 3 for i in range (3)]
        self.transitions[0][0] = self._make_cell(block, inplanes*1, midplanes*1, inplanes*1)  # 32x32x16
        self.transitions[1][1] = self._make_cell(block, inplanes*2, midplanes*2, inplanes*2)  # 16x16x32
        self.transitions[2][2] = self._make_cell(block, inplanes*4, midplanes*4, inplanes*4)  #   8x8x64
        self.transitions[0][1] = self._make_transition(down_block, inplanes, inplanes*2, 1)
        # self.transitions[0][2] = self._make_transition(down_block, inplanes, inplanes*4, 2)
        self.transitions[1][2] = self._make_transition(down_block, inplanes*2, inplanes*4, 1)
        # self.transitions[1][0] = self._make_transition(up_block, inplanes*2, inplanes, 1)
        # self.transitions[2][0] = self._make_transition(up_block, inplanes*4, inplanes, 2)
        # self.transitions[2][1] = self._make_transition(up_block, inplanes*4, inplanes*2, 1)

        self.layer = self._make_layer(self.transitions, inplanes)

        self.bn = nn.BatchNorm2d(inplanes*4*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(inplanes*4*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # todo
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            #elif isinstance(m, nn.BatchNorm2d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()
        
        if wnorm:
            self.wnorm()

    def _make_cell(self, block, inplanes, midplanes, planes):
        return block(inplanes, midplanes, planes, norm_func=self.norm_func)
    
    def _make_transition(self, block, inplanes, planes, layers=2):
        return block(inplanes, planes, layers=layers, norm_func=self.norm_func)

    def _make_layer(self, transitions, stride=1):
        return Parallel(transitions)

    def wnorm(self):
        for row in self.transitions:
            for elem in row:
                if elem is not None:
                    elem.wnorm()

    def reset(self):
        for row in self.transitions:
            for elem in row:
                if elem is not None:
                    elem.reset()

    def copy(self, func):
        self.layer.copy(func.layer)

    def _get_diffs(self):
        info = self.layer.get_diffs()
        diffs = [info]
        return diffs

    def forward(self, x, debug=False):
        self.layer.reset()

        bs = x.shape[0]
        h, w = x.shape[2:]

        x = self.conv1(x)

        z1 = torch.zeros((bs, self.inplanes, h, w), device=x.device)
        z2 = torch.zeros((bs, self.inplanes * 2, h // 2, w // 2), device=x.device)
        z3 = torch.zeros((bs, self.inplanes * 4, h // 4, w // 4), device=x.device)

        self.reset()
        for i in range(self.layers):
            x, [z1, z2, z3] = self.layer(x, [z1, z2, z3], debug=debug)
        z = z3

        z = self.bn(z)
        z = self.relu(z)
        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        z = self.fc(z)

        if debug:
            return z, self.layer.get_diffs()
        
        return z


def wtii_preact_parresnet110_cifar(layers=18, **kwargs):
    model = WTIIPreAct_ParResNet_Cifar(PreActBasicParBlock, DownBlock, UpBlock, layers, **kwargs)
    return model


if __name__ == '__main__':
    net = wtii_preact_parresnet110_cifar(inplanes=32, midplanes=136, wnorm=False)
    #net = preact_resnet110_cifar()
    y, diffs = net(torch.randn(1, 3, 32, 32), debug=True)
    print(net)
    print(y.size())
    n_all_param = sum([p.nelement() for p in net.parameters() if p.requires_grad])
    print(f'#params = {n_all_param}')
    print(diffs)

