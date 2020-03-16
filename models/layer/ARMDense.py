import torch
import torch.nn as nn
import torch.nn.functional as F


class ARMDense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_decay=1e-4, lamba=0., droprate_init=0.5, k=7,
                 local_rep=True, init_size=-1, device='cpu'):
        super(ARMDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decay = weight_decay
        self.lamba = lamba
        self.weights = nn.Parameter(torch.Tensor(in_features, out_features, ))
        self.z_phi = nn.Parameter(torch.Tensor(in_features))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))

        self.droprate_init = droprate_init
        self.k = k
        self.u = None
        self.forward_mode = True
        self.local_rep = local_rep
        self.activated_neuron_size = init_size
        self.reset_parameters()
        self.device = device
        print(self)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weights, mode='fan_out')
        if self.activated_neuron_size == -1:
            self.z_phi.data.normal_(3 / self.k, 1e-2)
        else:
            print('{}:{}'.format(self.activated_neuron_size, self.in_features))
            self.z_phi.data[:self.activated_neuron_size].normal_(3 / self.k, 1e-2)
            self.z_phi.data[self.activated_neuron_size:].normal_(-40 / self.k, 1e-2)
        if self.use_bias:
            self.bias.data.fill_(0)

    def expand_layer(self, fill=0):
        if self.activated_neuron_size < self.in_features and self.activated_neurons() >= self.activated_neuron_size:
            self.z_phi.data[self.activated_neuron_size].normal_(fill / self.k, 1e-2)
            self.activated_neuron_size += 1

    def update_phi_gradient(self, f1, f2):
        e = self.k * ((f1 - f2) * (self.u - .5)).mean(dim=0)
        self.z_phi.grad = e

    def regularization(self):
        pi = torch.sigmoid(self.k * self.z_phi)
        l0 = self.lamba * pi.sum() * self.out_features
        logpw_col = torch.sum(.5 * self.weight_decay * self.weights.pow(2), 1)
        logpw = torch.sum(pi.data * logpw_col)
        logpb = 0 if not self.use_bias else torch.sum(.5 * self.weight_decay * self.bias.pow(2))
        l2 = logpw + logpb
        return l0 + l2

    def sample_z(self, batch_size):
        pi = torch.sigmoid(self.k * self.z_phi).detach()
        if self.forward_mode:
            z = torch.FloatTensor(batch_size, self.in_features).zero_().to(self.device)
            if self.training:
                if self.local_rep:
                    self.u = torch.FloatTensor(self.in_features).uniform_(0, 1).expand(batch_size, self.in_features).to(self.device)
                else:
                    self.u = torch.FloatTensor(batch_size, self.in_features).uniform_(0, 1).to(self.device)
                z[self.u < pi.expand(batch_size, self.in_features)] = 1
                self.train_z = z
            else:
                z = pi.expand(batch_size, self.in_features)
                z[z < .5] = 0
                self.test_z = z
        else:
            pi2 = 1 - pi
            if self.u is None:
                raise Exception('Forward pass first')
            z = torch.FloatTensor(self.u.size()).zero_().to(self.device)
            z[self.u > pi2.expand(batch_size, self.in_features)] = 1

        return z

    def forward(self, input):
        """ forward for fc """
        xin = input.mul(self.sample_z(input.size(0)))
        output = xin.mm(self.weights)
        if self.use_bias:
            output += self.bias
        return output

    def masked_weight(self):
        return self.weights * self.test_z[0].reshape(self.in_features, 1)

    def activated_neurons(self):
        return (self.test_z > 0).sum() / self.test_z.size(0)

    def expected_activated_neurons(self):
        return (self.train_z > 0).sum() / self.train_z.size(0)
