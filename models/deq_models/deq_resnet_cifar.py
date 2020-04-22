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
        self.grads = AverageMeter()
        self.diffs = AverageMeter()
        self.forward_tr = []

    def update(self, input):
        self.grads.update(calc_grad_norm(self), input.size(0))
        self.diffs.update(self.layer._diffs)
    
    def forward(self, x, f_thres=None, debug=False, pretraining=False):  
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
            self.layer._diffs = min_diff
            return z
        self.layer.reset()
        self.func_copy.copy(self.func)
        z1s = self.func.img2seq(z)
        us = self.func.img2seq(x)
        z1s = self.deq(z1s, us, None, threshold=f_thres, debug=debug)
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


    def get_grads(self):
        # just for plots
        return {
            0: self.deq_layer1.grads,
            1: self.deq_layer2.grads,
            2: self.deq_layer3.grads
        }
    
    def get_diffs(self):
        # just for plots
        return {
            0: self.deq_layer1.diffs,
            1: self.deq_layer2.diffs,
            2: self.deq_layer3.diffs
        }

    def update_meters(self, input):
        # just for plots
        self.deq_layer1.update(input)
        self.deq_layer2.update(input)
        self.deq_layer3.update(input)

    def forward(self, x, train_step=-1,f_thres=30, b_thres=40, debug=False):

        do_pretraning = 0 <= train_step < self.pretrain_steps or self.test_mode == "forward"
        x = self.conv1(x)
        x = self.deq_layer1(x, f_thres, debug, do_pretraning)
        x = self.down12(x)
        x = self.deq_layer2(x, f_thres, debug, do_pretraning)
        x = self.down23(x)
        x = self.deq_layer3(x, f_thres, debug, do_pretraning)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if not debug:
            return x
        
        if self.test_mode == "broyden":
            diffs = self._get_diffs()
            info = '\n'.join(diffs)
        else:
            diffs = [self.deq_layer1.forward_tr, self.deq_layer2.forward_tr, self.deq_layer3.forward_tr]
            info = {"layer" + str(i) : list(map(lambda x : f"{x:.4f}", x)) for i, x in enumerate(diffs)}
            info = "\n".join(map(str, info.items()))
        return x, info
        
def wtii_deq_preact_resnet110_cifar(**kwargs):
    model = DEQSeqResNet(IIPreActBasicBlock, [18, 18, 18], **kwargs)
    return model

if __name__=="__main__":
    net = wtii_deq_preact_resnet110_cifar(wnorm=False, inplanes=32 + 10)
    # net = preact_resnet110_cifar()
    # print(net)
    y, diffs = net(torch.randn(1, 3, 32, 32), debug=True, train_step=-1)
    print(y.size())
    n_all_param = sum([p.nelement() for p in net.parameters() if p.requires_grad])
    print(f'#params = {n_all_param}')
    print(diffs)

    y.mean().backward()