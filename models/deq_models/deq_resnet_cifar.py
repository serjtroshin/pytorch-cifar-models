import sys
sys.path.append("../../")
from models.deq_models.deq_modules.deq import *

from models.resnet_cifar import WTIIPreAct_ResNet_Cifar, IIPreActBasicBlock
from models.deq_models.deq_resnet_cifar_module import ParResNetDEQModule


class DEQParResNetLayer(nn.Module):
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

    def forward(self, z1ss, uss, z0, *args, **kwargs):
        debug=kwargs.get('debug', False)
        z = self.seq2img(z1ss)
        x = self.seq2img(uss)
        (z, x) = self.layer(z, x, debug=debug)
        z1ss = self.img2seq(z)
        uss = self.img2seq(x)
        return z1ss





class DEQParResNet(WTIIPreAct_ResNet_Cifar):

    def __init__(self, block, layers, num_classes=10, **kwargs):
        super(DEQParResNet, self).__init__(block, layers, num_classes=num_classes, copy_layers=True, **kwargs)

        self.pretrain_steps = kwargs.get("pretrain_steps", 200)
        self.n_layer = kwargs.get("n_layer", 3)

        self.func1 = DEQParResNetLayer(self.layer1)
        self.func_copy1 = DEQParResNetLayer(self.layer1_copy)
        self.func2 = DEQParResNetLayer(self.layer2)
        self.func_copy2 = DEQParResNetLayer(self.layer2_copy)
        self.func3 = DEQParResNetLayer(self.layer3)
        self.func_copy3 = DEQParResNetLayer(self.layer3_copy)

        self.deq1 = ParResNetDEQModule(self.func1, self.func_copy1)
        self.deq2 = ParResNetDEQModule(self.func2, self.func_copy2)
        self.deq3 = ParResNetDEQModule(self.func3, self.func_copy3)

    def forward(self, x, train_step=-1,f_thres=30, b_thres=40, debug=False):
        if 0 <= train_step < self.pretrain_steps:
            if debug: print("FORWARD")
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

        else:
            if debug: print("DEQ")
            x = self.conv1(x)
            z = torch.zeros_like(x)
            (z, x) = self.down01(z, x)  # TODO test if we need this
            # layer1
            self.layer1.reset()
            z1s = self.func1.img2seq(z)
            us = self.func1.img2seq(x)
            self.layer1.reset()
            z1s = self.deq1(z1s, us, None, threshold=f_thres, debug=debug)
            z = self.func1.seq2img(z1s)
            # layer2
            (z, x) = self.down12(z, x)
            x = z
            z = torch.zeros_like(x)
            z1s = self.func2.img2seq(z)
            us = self.func2.img2seq(x)
            self.layer2.reset()
            z1s = self.deq2(z1s, us, None, threshold=f_thres, debug=debug)
            z = self.func2.seq2img(z1s)
            # layer3
            (z, x) = self.down23(z, x)
            x = z
            z = torch.zeros_like(x)
            z1s = self.func3.img2seq(z)
            us = self.func3.img2seq(x)
            self.layer3.reset()
            z1s = self.deq3(z1s, us, None, threshold=f_thres, debug=debug)
            z = self.func3.seq2img(z1s)

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
        
def wtii_deq_preact_resnet110_cifar(**kwargs):
    model = DEQParResNet(IIPreActBasicBlock, [18, 18, 18], **kwargs)
    return model

if __name__=="__main__":
    net = wtii_deq_preact_resnet110_cifar(wnorm=False, inplanes=32 + 10)
    # net = preact_resnet110_cifar()
    print(net)
    y, diffs = net(torch.randn(1, 3, 32, 32), debug=True, train_step=-1)
    print(y.size())
    n_all_param = sum([p.nelement() for p in net.parameters() if p.requires_grad])
    print(f'#params = {n_all_param}')
    print(diffs)

    y.mean().backward()
