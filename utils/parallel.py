import torch
from torch import nn
from .plot_utils import ConvergenceMeter


class Parallel(nn.Module):
    def __init__(self, transitions):
        super(Parallel, self).__init__()
        self.transitions = transitions
        for i in range(len(transitions)):
            for j in range(len(transitions[0])):
                setattr(self, f'transition_f_{i}_{j}', transitions[i][j])
    def forward(self, x, zs, channels_dim=1):
        assert len(zs) == len(self.transitions), "Number of states must match the number of cells"
        assert len(self.transitions) == len(self.transitions[0]), "Dimentions must match"
        zs_next = []
        for i in range(len(zs)):
            z_input = None
            z_other = []
            for j, zj in enumerate(zs):
                if j == i: 
                    z_input = zj
                    continue
                #print("-"*90)
                #print("transition", j, i)
                #print(zj.shape)
                #print(self.transitions[j][i])
                z_other.append(self.transitions[j][i](zj))
                #print(z_other[-1].shape)
                #print("-"*90)
            if i == 0:
                #print("shape", list(map(lambda x: x.shape, z_other + [x])))
                z_other = torch.cat(z_other + [x], dim=channels_dim)
            else:
                z_other = torch.cat(z_other, dim=channels_dim)
            #print("-"*90)
            #print(z_input.shape, z_other.shape)
            #print(self.transitions[i][i])
            #print("^"*90)
            z_next = self.transitions[i][i](z_input, z_other)
            zs_next.append(z_next)
        return x, zs_next

