from copy import deepcopy

import torch
import torch.nn as nn

from models.layer import ArmConv2d
from models.layer import ARMDense

import numpy as np

def get_flat_fts(in_size, fts):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    f = fts(torch.autograd.Variable(dummy_input))
    return int(np.prod(f.size()[1:]))


class NPNLeNet5(nn.Module):
    def __init__(self, input_size=(1, 28, 28), param=(20, 50, 500), init_size=(20, 50, 500), k=7,
                 N=60000, weight_decay=0.0005, lambas=(0, 0, 0, 0), device='cpu'):
        super(NPNLeNet5, self).__init__()
        self.N = N
        self.weight_decay = weight_decay

        convs = [ArmConv2d(input_size[0], param[0], 5, droprate_init=0.5, k=k, lamba=lambas[0], local_rep=True,
                           weight_decay=self.weight_decay, init_size=init_size[0], device=device),
                 nn.ReLU(), nn.MaxPool2d(2),
                 ArmConv2d(param[0], param[1], 5, droprate_init=0.5, k=k, lamba=lambas[1], local_rep=True,
                           weight_decay=self.weight_decay, init_size=init_size[1], device=device),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            self.convs = self.convs.cuda()

        flat_fts = get_flat_fts(input_size, self.convs)
        fcs = [ARMDense(flat_fts, param[2], droprate_init=0.5, k=k, lamba=lambas[2], local_rep=True,
                        weight_decay=self.weight_decay, init_size=flat_fts * init_size[1] // param[1], device=device),
               nn.ReLU(),
               ARMDense(param[2], 10, droprate_init=0.5, k=k, lamba=lambas[3], local_rep=True,
                        weight_decay=self.weight_decay, init_size=init_size[2], device=device)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, ARMDense) or isinstance(m, ArmConv2d):
                self.layers.append(m)

    def update_phi_gradient(self, f1, f2):
        for layer in self.layers:
            layer.update_phi_gradient(f1, f2)

    def forward_mode(self, mode):
        for layer in self.layers:
            layer.forward_mode = mode

    def score(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        o = self.fcs(o)
        return o

    def forward(self, x, y=None):
        if self.training:
            self.forward_mode(True)
            score = self.score(x)

            # update grad of phi
            if y is not None:
                self.eval()
                self.forward_mode(False)
                score2 = self.score(x).data
                f1 = nn.CrossEntropyLoss()(score2, y).data
                f2 = nn.CrossEntropyLoss()(score, y).data

                self.update_phi_gradient(f1, f2)
            self.train()
        else:
            self.forward_mode(True)
            score = self.score(x)
        return score

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += (1. / self.N) * layer.regularization()
        return regularization

    def set_k(self, k):
        self.k = k
        for layer in self.layers:
            layer.k = k

    def get_activated_neurons(self):
        return [layer.activated_neurons() for layer in self.layers]

    def z_phis(self):
        return [layer.z_phi for layer in self.layers]

    def expand(self, fill=0):
        for l in self.layers:
            l.expand_layer(fill)

    def para_num(self):
        l = [layer.activated_neurons().cpu().numpy() for layer in self.layers]
        return l[0] * 25.0 + l[1] * l[0] * 25.0 + l[2] * l[3] + l[3] * 10.0
