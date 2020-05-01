import sys
sys.path.append("../../")
from models.deq_models.deq_modules.deq import *

from models.resnet_cifar import WTIIPreAct_ResNet_Cifar, IIPreActBasicBlock
from models.deq_models.deq_resnet_cifar_module import ParResNetDEQModule
from utils.plot_utils import AverageMeter, calc_grad_norm


class DEQParResNetLayer(nn.Module):
    """
        Implements DEQ func
    """
    def __init__(self, layer, **kwargs):
        super(DEQParResNetLayer, self).__init__()
        self.layer = layer

    def update_shapes(self, shapes):
        self.shapes = shapes

    def seq2img(self, z1ss):
        """
            1, 16, 32 * 32 -> 1, 16, 32, 32
        """
        bs, ch, hw = z1ss.shape
        h = int(round(hw**0.5))
        w = h
        z = z1ss.reshape((bs, ch, h, w))
        return z

    def img2seq(self, z):
        """
            1, 16, 32, 32 -> 1, 16, 32 * 32
        """
        bs, ch, h, w = z.shape
        z1ss = z.reshape((bs, ch, h * w))
        return z1ss

    def copy(self, func):
        self.layer.copy(func.layer)
        
    def forward(self, z1ss, uss, z0, *args, **kwargs):
        debug=kwargs.get('debug', False)
        z = self.seq2img(z1ss)
        x = self.seq2img(uss)
        (z, x) = self.layer(z, x, debug=debug)
        z1ss = self.img2seq(z)
        uss = self.img2seq(x)
        return z1ss


class DEQMeter(object):
    def __init__(self, layer):
        self.grads = AverageMeter()
        self.forward_diffs = AverageMeter()
        self.backward_diffs = AverageMeter()
        self.pretrain_diffs = AverageMeter()
        self.layer = layer

    def update(self, input):
        self.grads.update(calc_grad_norm(self.layer), input.size(0))  
        self.forward_diffs.update(self.layer.info["forward_diffs"])
        self.backward_diffs.update(self.layer.info["backward_diffs"])
        self.pretrain_diffs.update(self.layer.info["pretrain_diffs"])


class ResNetToDEQWrapper(nn.Module):
    """
    This class is just to reduce code
    """
    def __init__(self, layer, layer_copy, n_layer=3):
        super(ResNetToDEQWrapper, self).__init__()
        self.layer = layer
        self.func = DEQParResNetLayer(self.layer)
        self.func_copy = DEQParResNetLayer(layer_copy)
        self.deq = ParResNetDEQModule(self.func, self.func_copy)
        self.n_layer = n_layer

        self.meter = DEQMeter(self.layer)
        
        self.forward_tr = []

    def update(self, input):
        self.meter.update(input)
    
    def forward(self, x, f_thres=None, debug=False, pretraining=False, store_trajs=None):  
        z = torch.zeros_like(x)
        if debug:
            self.forward_tr = []
        if pretraining:
            self.layer.reset()
            min_diff = 1e10
            prev = z
            for i in range(self.n_layer):
                z, _ = self.layer(z, x, debug=debug)
                if debug:
                    self.forward_tr.append((z - prev).norm().item())
                min_diff = min(min_diff, (z - prev).norm().item())
                prev = z
            self.layer.info["pretrain_diffs"] = min_diff
            return z
        self.layer.reset()
        self.func_copy.copy(self.func)
        z1s = self.func.img2seq(z)
        us = self.func.img2seq(x)
        z1s = self.deq(z1s, us, None, threshold=f_thres, debug=debug, store_trajs=store_trajs)
        z = self.func.seq2img(z1s)
        return z




class DEQSeqResNet(WTIIPreAct_ResNet_Cifar):
    """
    Sequential stacking of DEQ layers.
    """

    def __init__(self, block, layers, num_classes=10, **kwargs):
        super(DEQSeqResNet, self).__init__(block, layers, num_classes=num_classes, copy_layers=True, **kwargs)

        self.pretrain_steps = kwargs.get("pretrain_steps", 200)
        self.n_layer = kwargs.get("n_layer", 3)
        self.test_mode = kwargs.get("test_mode", "broyden")

        self.deq_layer1 = ResNetToDEQWrapper(self.layer1, self.layer1_copy, n_layer=self.n_layer)
        self.deq_layer2 = ResNetToDEQWrapper(self.layer2, self.layer2_copy, n_layer=self.n_layer)
        self.deq_layer3 = ResNetToDEQWrapper(self.layer3, self.layer3_copy, n_layer=self.n_layer)

        print('=' * 100)
        n_all_params_layer1 = sum([p.nelement() for p in self.layer1.parameters() if p.requires_grad])
        print(f'#true params layer1 = {n_all_params_layer1}')


    def get_grads(self):
        # just for plots
        return {
            0: self.deq_layer1.meter.grads,
            1: self.deq_layer2.meter.grads,
            2: self.deq_layer3.meter.grads
        }
    
    def get_diffs(self):
        # just for plots
        return {
            'forward_diffs':
                {
                    0: self.deq_layer1.meter.forward_diffs,
                    1: self.deq_layer2.meter.forward_diffs,
                    2: self.deq_layer3.meter.forward_diffs
                },
            'backward_diffs':
                {
                    0: self.deq_layer1.meter.backward_diffs,
                    1: self.deq_layer2.meter.backward_diffs,
                    2: self.deq_layer3.meter.backward_diffs
                },
            'pretrain_diffs':
                {
                    0: self.deq_layer1.meter.pretrain_diffs,
                    1: self.deq_layer2.meter.pretrain_diffs,
                    2: self.deq_layer3.meter.pretrain_diffs
                }
        }

    def update_meters(self, input):
        # just for plots
        self.deq_layer1.update(input)
        self.deq_layer2.update(input)
        self.deq_layer3.update(input)

    def forward(self, x, train_step=-1,f_thres=30, b_thres=40, debug=False, store_trajs=None):

        do_pretraning = 0 <= train_step < self.pretrain_steps or self.test_mode == "forward"
        x = self.conv1(x)
        x = self.deq_layer1(x, f_thres, debug, do_pretraning, store_trajs)
        x = self.down12(x)
        # x = self.deq_layer2(x, f_thres, debug, do_pretraning, store_trajs)
        x = self.down23(x)
        # x = self.deq_layer3(x, f_thres, debug, do_pretraning, store_trajs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if not debug:
            return x
        
        diffs = self._get_diffs()
        info = '\n'.join(diffs)
        return x, info
        
def wtii_deq_preact_resnet110_cifar(**kwargs):
    model = DEQSeqResNet(IIPreActBasicBlock, [18, 18, 18], **kwargs)
    return model

if __name__=="__main__":
    net = wtii_deq_preact_resnet110_cifar(wnorm=True, inplanes=32 + 10)
    # net = preact_resnet110_cifar()
    # print(net)
    y, diffs = net(torch.randn(1, 3, 32, 32), debug=True, train_step=-1)
    print(y.size())
    n_all_param = sum([p.nelement() for p in net.parameters() if p.requires_grad])
    print(f'#params = {n_all_param}')
    print(diffs)

    y.mean().backward()
