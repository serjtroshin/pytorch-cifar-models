import sys
sys.path.append("../../")
from models.deq_models.deq_modules.deq import *

from models.par_resnet_cifar import *
from models.deq_models.deq_par_resnet_cifar_module import ParResNetDEQModule
from utils.plot_utils import DEQMeter


class DEQParResNetLayer(nn.Module):
    def __init__(self, layer, **kwargs):
        super(DEQParResNetLayer, self).__init__()
        self.layer = layer

    def update_shapes(self, shapes):
        self.shapes = shapes

    def seq2img(self, z1ss):
        bs, ch, hw = z1ss.shape
        z1sh, z2sh, z3sh = self.shapes
        z1 = z1ss[:, :, :z1sh[2]*z1sh[3]].reshape((bs, ch, z1sh[2], z1sh[3]))  # 16, 32, 32
        #assert z1.shape == (bs, 16, 32, 32)
        z2 = z1ss[:, :, z1sh[2]*z1sh[3]:  z1sh[2]*z1sh[3] + z2sh[2]*z2sh[3]*2].reshape(bs, ch * 2, z2sh[2], z2sh[3]) # 32, 16, 16
        #assert z2.shape == (bs, 32, 16, 16)
        z3 = z1ss[:, :, z1sh[2]*z1sh[3] + z2sh[2]*z2sh[3]*2:].reshape(bs, ch * 4, z3sh[2], z3sh[3]) # 64, 8, 8
        #assert z3.shape == (bs, 64, 8, 8)
        return z1, z2, z3

    def img2seq(self, z1, z2, z3):
        z1sh, z2sh, z3sh = self.shapes
        bs, ch, _, _ = z1sh
        z1ss = torch.cat([  z1.reshape(bs, ch, z1sh[2]*z1sh[3]), 
                            z2.reshape(bs, ch, z2sh[2]*z2sh[3]*2), 
                            z3.reshape(bs, ch, z3sh[2]*z3sh[3]*4)  ], dim=-1)
        return z1ss

    def copy(self, func):
        self.layer.copy(func.layer)

    def forward(self, z1ss, uss, z0, *args, **kwargs):
        debug=kwargs.get('debug', False)
        x = uss
        z1, z2, z3 = self.seq2img(z1ss)
        x, [z1, z2, z3] = self.layer(x, [z1, z2, z3], debug=debug)
        z1ss = self.img2seq(z1, z2, z3)
        return z1ss


class DEQParResNet(WTIIPreAct_ParResNet_Cifar):

    def __init__(self, block, down_block, up_block, layers, num_classes=10, **kwargs):
        super(DEQParResNet, self).__init__(block, down_block, up_block, layers, num_classes=10, **kwargs)

        self.pretrain_steps = kwargs.get("pretrain_steps", -1)
        self.n_layer = kwargs.get("n_layer", 3)

        self.func = DEQParResNetLayer(self.layer)
        self.func_copy = copy.deepcopy(self.func)

        self.deq = ParResNetDEQModule(self.func, self.func_copy)

        self.meter = DEQMeter(self.layer)

    def _infer_shapes(self, x):
        bs = x.shape[0]
        h, w = x.shape[2:]
        return [(bs, self.inplanes, h, w), (bs, self.inplanes * 2, h // 2, w // 2), (bs, self.inplanes * 4, h // 4, w // 4)]

    def update_meters(self, input):
        # just for plots
        self.meter.update(input)

    def get_grads(self):
        # just for plots
        return {
            0: self.meter.grads,
            1: self.meter.grads,
            2: self.meter.grads
        }
    
    def get_diffs(self):
        # just for plots
        return {
            'forward_diffs':
                {
                    0: self.meter.forward_diffs,
                },
            'backward_diffs':
                {
                    0: self.meter.backward_diffs,
                },
            'pretrain_diffs':
                {
                    0: self.meter.pretrain_diffs,
                }
        }

    def forward(self, x, train_step=-1,f_thres=30, b_thres=40, debug=False, store_trajs=None):
        self.layer.reset()

        bs = x.shape[0]
        h, w = x.shape[2:]

        us = self.conv1(x)
        shapes = self._infer_shapes(us)
        self.func.update_shapes(shapes)
        self.func_copy.update_shapes(shapes)

        z1 = torch.zeros((bs, self.inplanes, h, w), device=us.device)
        z2 = torch.zeros((bs, self.inplanes * 2, h // 2, w // 2), device=us.device)
        z3 = torch.zeros((bs, self.inplanes * 4, h // 4, w // 4), device=us.device)

        # self.reset()
        z = self.func.img2seq(z1, z2, z3)

        if 0 <= train_step < self.pretrain_steps:
            if debug:
                self.forward_tr = []
            self.layer.reset()
            min_diff = 1e10
            prev = z
            for i in range(self.n_layer):
                z = self.func(z, us, None, debug=debug)
                if debug:
                    self.forward_tr.append((z - prev).norm().item())
                min_diff = min(min_diff, (z - prev).norm().item())
                prev = z
            self.layer.info["pretrain_diffs"] = min_diff
        else:
            self.layer.reset()
            self.func_copy.copy(self.func)
            z = self.deq(z, us, None, threshold=f_thres, debug=debug, store_trajs=store_trajs)

        _, _, z3 = self.func.seq2img(z)

        z = z3

        z = self.bn(z)
        z = self.relu(z)
        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        
        if not debug:
            return z
        
        diffs = self._get_diffs()
        info = '\n'.join(diffs)
        return z, info
        
def deq_parresnet110_cifar(layers=18, **kwargs):
    model = DEQParResNet(PreActBasicParBlock, DownBlock, UpBlock, layers, **kwargs)
    return model

if __name__=="__main__":
    net = deq_parresnet110_cifar(18, pretrain_steps=10, n_layer=3, inplanes=61)
    y, diffs = net(torch.randn(1, 3, 32, 32), debug=True, train_step=-1)
    print(y.size())

    del net.func_copy  # only a copy of parameters (not nes)
    del net.deq
    n_all_param = sum([p.nelement() for p in net.parameters() if p.requires_grad])
    print(f'#params = {n_all_param}')
    print(diffs)

    # y.mean().backward()