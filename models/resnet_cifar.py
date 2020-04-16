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

sys.path.append('../')
from utils.utils import SequentialLayer
from utils.optimization import weight_norm, VariationalDropout


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def get_norm_func():
    return {
        "inst" : nn.InstanceNorm2d,
        "batch" : nn.BatchNorm2d
    }

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IIPreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                    norm_func=nn.InstanceNorm2d, identity_mapping=False,
                    dropout=0.0):
        super(IIPreActBasicBlock, self).__init__()
        self.dropout = VariationalDropout(dropout) # input dropout
        self.bn1 = norm_func(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes * 2, planes, stride)
        self.bn2 = norm_func(planes)
        self.conv2 = conv3x3(planes, planes)
        if not identity_mapping:
            self.norml = norm_func(planes)
        self.downsample = downsample
        self.stride = stride

        self.identity_mapping = identity_mapping

    def wnorm(self):
        self.conv1, self.conv1_fn = weight_norm(module=self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(module=self.conv2, names=['weight'], dim=0)

    def reset(self):
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)

    def copy(self, func):
        self.conv1.weight.data = func.conv1.weight.data.clone()
        self.conv2.weight.data = func.conv2.weight.data.clone()

    def forward(self, z, x):
        x = self.dropout(x)

        residual = z
        
        z = self.bn1(z)
        z = self.relu(z)

        z_cat = torch.cat((z, x), 1)
        if self.downsample is not None:
            residual = self.downsample(z)
            x = self.downsample(x)
        
        out = self.conv1(z_cat)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        if not self.identity_mapping:
            out = self.norml(out) 

        return out, x


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class WTIIPreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, copy_layers=False, **kwargs):
        super(WTIIPreAct_ResNet_Cifar, self).__init__()

        norm_func=kwargs.get("norm_func", "inst")
        wnorm=kwargs.get("wnorm", False) # weight normalization
        self.identity_mapping=kwargs.get("identity_mapping", False) # is identity path clear
        self.inplanes=kwargs.get("inplanes", 16)
        self.dropout=kwargs.get("dropout", 0.0)
        inplanes = self.inplanes

        self.norm_func = get_norm_func()[norm_func]

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        tmp = self.inplanes
        self.down01, self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.down12, self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=2)
        self.down23, self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=2)
        if copy_layers:
            self.inplanes = tmp
            _, self.layer1_copy = self._make_layer(block, inplanes, layers[0])
            _, self.layer2_copy = self._make_layer(block, inplanes*2, layers[1], stride=2)
            _, self.layer3_copy = self._make_layer(block, inplanes*4, layers[2], stride=2)
        self.bn = self.norm_func(inplanes*4*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(inplanes*4*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            #elif isinstance(m, nn.BatchNorm2d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()

        if wnorm:
          self.wnorm()

    def wnorm(self):
        # wnorm
        raise NotImplemented # update for deq func_copy
        self.down01.wnorm()
        self.down12.wnorm()
        self.down23.wnorm()
        self.layer1.wnorm()
        self.layer2.wnorm()
        self.layer3.wnorm()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            norm_func=self.norm_func, 
                            identity_mapping=self.identity_mapping,
                            dropout=self.dropout))
        self.inplanes = planes*block.expansion

        layers.append(SequentialLayer(
                        block(self.inplanes, planes, 
                            norm_func=self.norm_func, 
                            identity_mapping=self.identity_mapping,
                            dropout=self.dropout)))
        layers[-1].layers = blocks - 1
            
        return layers

    def forward(self, x, debug=False, **kwargs):

        x = self.conv1(x)
        z = torch.zeros_like(x)
        (z, x) = self.down01(z, x)  # TODO test if we need this

        self.layer1.reset()
        for i in range(self.layer1.layers):
            (z, x) = self.layer1(z, x, debug=debug)
        
        (z, x) = self.down12(z, x)
        x = z
        z = torch.zeros_like(x)
        self.layer2.reset()
        for i in range(self.layer2.layers):
            (z, x) = self.layer2(z, x, debug=debug)

        (z, x) = self.down23(z, x)
        x = z
        z = torch.zeros_like(x)
        self.layer3.reset()
        for i in range(self.layer3.layers):
            (z, x) = self.layer3(z, x, debug=debug)
        #print(f"block 3 out: {z.shape}")

        z = self.bn(z)
        z = self.relu(z)
        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        
        if not debug:
            return z
        info1 = self.layer1.get_diffs()
        info2 = self.layer2.get_diffs()
        info3 = self.layer3.get_diffs()
        diffs = [info1, info2, info3]
        info = {"layer" + str(i) : list(map(lambda x : f"{x:.4f}", x)) for i, x in enumerate(diffs)}
        info = "\n".join(map(str, info.items()))
        return z, info



def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def wtii_preact_resnet110_cifar(**kwargs):
    model = WTIIPreAct_ResNet_Cifar(IIPreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = wtii_preact_resnet110_cifar(wnorm=True, inplanes=32 + 10)
    # net = preact_resnet110_cifar()

    y, diffs = net(torch.randn(1, 3, 32, 32), debug=True)
    print(net)
    print(y.size())
    n_all_param = sum([p.nelement() for p in net.parameters() if p.requires_grad])
    print(f'#params = {n_all_param}')
    print(diffs)

