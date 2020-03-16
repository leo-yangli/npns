import torch.nn as nn
from models.layer.ARMDense import ARMDense


class NPNMLP(nn.Module):
    def __init__(self, input_dim=100, num_classes=2, param=(80), init_size=(3,3), k=1, N=8160,
                 weight_decay=5e-4, lambas=(.3, 1), device='cpu'):
        # spa .3, 1
        # exp .35, 14
        super(NPNMLP, self).__init__()

        self.layer_dims = param
        self.input_dim = input_dim
        self.N = N
        self.weight_decay = weight_decay
        self.lambas = lambas
        self.k = k

        layers = [ARMDense(100, 80, droprate_init=0.1, weight_decay=self.weight_decay,
                            lamba=lambas[0], local_rep=True, init_size=init_size[0], device=device), nn.ReLU(),
                  ARMDense(80, num_classes, droprate_init=0.1, weight_decay=self.weight_decay,
                           lamba=lambas[1], local_rep=True, init_size=init_size[1], device=device)]

        self.output = nn.Sequential(*layers)
        self.layers = []
        for m in self.modules():
            if isinstance(m, ARMDense):
                self.layers.append(m)

    def score(self, x):
        return self.output(x.view(-1, self.input_dim))

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

    def update_phi_gradient(self, f1, f2):
        for layer in self.layers:
            layer.update_phi_gradient(f1, f2)

    def forward_mode(self, mode):
        for layer in self.layers:
            layer.forward_mode = mode

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += (1. / self.N) * layer.regularization()
        return regularization

    def get_activated_neurons(self):
        return [layer.activated_neurons() for layer in self.layers]

    def get_expected_activated_neurons(self):
        return [layer.expected_activated_neurons() for layer in self.layers]

    def prune_rate(self):
        l = [layer.activated_neurons().cpu().numpy() for layer in self.layers]
        return 100 - 100.0 * (l[0]*l[1]+l[1]*2.) / (100.*80+160)

    def z_phis(self):
        return [layer.z_phi for layer in self.layers]

    def para_num(self):
        l = [layer.activated_neurons().cpu().numpy() for layer in self.layers]
        return l[0] * l[1] + l[1] * 2

    def expand(self, fill):
        for l in self.layers:
            l.expand_layer(fill)

    def set_k(self, k):
        self.k = k
        for layer in self.layers:
            layer.k = k
