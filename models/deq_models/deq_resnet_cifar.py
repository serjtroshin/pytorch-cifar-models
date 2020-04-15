import sys
sys.path.append("../../")
from models.deq_models.deq_modules.deq import *

from models.par_resnet_cifar import *
from models.deq_models.deq_par_resnet_cifar_module import ParResNetDEQModule


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
        assert z1.shape == (bs, 16, 32, 32)
        z2 = z1ss[:, :, z1sh[2]*z1sh[3]:  z1sh[2]*z1sh[3] + z2sh[2]*z2sh[3]*2].reshape(bs, ch * 2, z2sh[2], z2sh[3]) # 32, 16, 16
        assert z2.shape == (bs, 32, 16, 16)
        z3 = z1ss[:, :, z1sh[2]*z1sh[3] + z2sh[2]*z2sh[3]*2:].reshape(bs, ch * 4, z3sh[2], z3sh[3]) # 64, 8, 8
        assert z3.shape == (bs, 64, 8, 8)
        return z1, z2, z3

    def img2seq(self, z1, z2, z3):
        z1sh, z2sh, z3sh = self.shapes
        bs, ch, _, _ = z1sh
        #print(z1sh, z2sh, z3sh)
        #print(list(map(lambda x: x.shape, [z1.reshape(bs, ch, z1sh[2]*z1sh[3]), z2.reshape(bs, ch, z2sh[2]*z2sh[3]*2), z3.reshape(bs, ch, z3sh[2]*z3sh[3]*4)])))
        z1ss = torch.cat([  z1.reshape(bs, ch, z1sh[2]*z1sh[3]), 
                            z2.reshape(bs, ch, z2sh[2]*z2sh[3]*2), 
                            z3.reshape(bs, ch, z3sh[2]*z3sh[3]*4)  ], dim=-1)
        return z1ss

    def forward(self, z1ss, uss, z0, *args, **kwargs):
        debug=kwargs.get('debug', False)
        x = uss
        z1, z2, z3 = self.seq2img(z1ss)
        x, [z1, z2, z3] = self.layer(x, [z1, z2, z3], debug=debug)
        z1ss = self.img2seq(z1, z2, z3)
        return z1ss





class DEQParResNet(WTIIPreAct_ResNet_Cifar):

    def __init__(self, block, layers, num_classes=10, **kwargs):
        super(DEQParResNet, self).__init__(self, block, layers, num_classes=10, **kwargs)

        self.pretrain_steps = kwargs.get("pretrain_steps", -1)
        self.n_layer = kwargs.get("n_layer", 3)

        self.func = DEQParResNetLayer(self.layer)
        self.func_copy = copy.deepcopy(self.func)

        self.deq = ParResNetDEQModule(self.func, self.func_copy)

    def _infer_shapes(self, x):
        bs = x.shape[0]
        h, w = x.shape[2:]
        return [(bs, self.inplanes, h, w), (bs, self.inplanes * 2, h // 2, w // 2), (bs, self.inplanes * 4, h // 4, w // 4)]

    def forward(self, x, train_step=-1,f_thres=30, b_thres=40, debug=False):
        self.layer.reset()

        bs = x.shape[0]
        h, w = x.shape[2:]

        x = self.conv1(x)
        shapes = self._infer_shapes(x)
        self.func.update_shapes(shapes)
        self.func_copy.update_shapes(shapes)

        z1 = torch.zeros((bs, self.inplanes, h, w), device=x.device)
        z2 = torch.zeros((bs, self.inplanes * 2, h // 2, w // 2), device=x.device)
        z3 = torch.zeros((bs, self.inplanes * 4, h // 4, w // 4), device=x.device)

        # self.reset()
        z1s = self.func.img2seq(z1, z2, z3)

        if 0 <= train_step < self.pretrain_steps:
            for i in range(self.n_layer):
                z1s = self.func(z1s, x, None, debug=debug)
        else:
            z1s = self.deq(z1s, x, None, threshold=f_thres, debug=debug)

        _, _, z3 = self.func.seq2img(z1s)

        z = z3

        z = self.bn(z)
        z = self.relu(z)
        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        z = self.fc(z)

        if debug:
            if len(self.layer.get_diffs()) == 0:
                return z, self.layer._result_info
            diffs = self.layer.get_diffs()
            info = {"layer" + str(i) : list(map(lambda x : f"{x:.4f}", x)) for i, x in enumerate(diffs)}
            debug_info = "\n".join(map(str, info.items()))
            return z, debug_info
        
        return z
        
def deq_parresnet110_cifar(layers=18, **kwargs):
    model = DEQParResNet(PreActBasicParBlock, DownBlock, UpBlock, layers, **kwargs)
    return model

if __name__=="__main__":
    net = deq_parresnet110_cifar(18, pretrain_steps=-1, n_layer=3)
    x = torch.randn(1, 3, 32, 32)
    y, debug_info = net(x, train_step=-1, debug=True, f_thres=50)

    print(debug_info)