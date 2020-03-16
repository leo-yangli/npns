import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair as pair
import torch.nn.functional as F

class ArmConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 weight_decay=1.e-4, lamba=0.1/6e5, droprate_init=0.01, k=7, local_rep=True, init_size=-1, device='cpu'):
        super(ArmConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.weight_decay = weight_decay
        self.lamba = lamba
        self.k = k
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        self.weights = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.z_phi = Parameter(torch.Tensor(out_channels))
        self.dim_z = out_channels
        self.input_shape = None
        self.u = torch.Tensor(self.dim_z).uniform_(0, 1)
        self.droprate_init = droprate_init
        self.forward_mode = True
        self.local_rep = local_rep
        self.activated_neuron_size = init_size
        self.device = device
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weights, mode='fan_out')
        if self.activated_neuron_size == -1:
            # self.z_phi.data.fill_(114.5/self.k)
            self.z_phi.data.normal_(3/7, 1e-2)
        else:
            print('{}:{}'.format(self.activated_neuron_size, self.dim_z))
            self.z_phi.data[:self.activated_neuron_size].normal_(3 / self.k, 1e-2 / self.k)
            self.z_phi.data[self.activated_neuron_size:].normal_(-40 / self.k, 1e-2 / self.k)
        if self.use_bias:
            self.bias.data.fill_(0)

    def expand_layer(self, fill):
        if self.activated_neuron_size < self.out_channels and self.activated_neurons() >= self.activated_neuron_size:
            self.z_phi.data[self.activated_neuron_size].normal_(fill / self.k, 1e-2)
            # self.z_phi.data[self.activated_neuron_size].normal_(2.1 / self.k, 1e-2)
            self.activated_neuron_size += 1

    def update_phi_gradient(self, f1, f2):
        # only handle first part of phi's gradient
        e = self.k * ((f1 - f2) * (self.u - .5)).mean(dim=0)
        self.z_phi.grad = e

    def regularization(self):
        pi = torch.sigmoid(self.k * self.z_phi)
        l0 = self.lamba * pi.sum() * self.weights.view(-1).size()[0] / self.weights.size(0)
        wd_col = .5 * self.weight_decay * self.weights.pow(2).sum(3).sum(2).sum(1)
        wd = torch.sum(pi.data * wd_col)
        wb = 0 if not self.use_bias else torch.sum(pi * (.5 * self.weight_decay * self.bias.pow(2)))
        l2 = wd + wb
        return l0 + l2

    def sample_z(self, batch_size):
        pi = torch.sigmoid(self.k * self.z_phi).detach()

        if self.forward_mode:
            z = torch.FloatTensor(batch_size, self.dim_z).zero_().to(self.device)
            if self.training:
                if self.local_rep:
                    self.u = torch.FloatTensor(self.dim_z).uniform_(0, 1).expand(batch_size, self.dim_z).to(self.device)
                else:
                    self.u = torch.FloatTensor(batch_size, self.dim_z).uniform_(0, 1).to(self.device)

                z[self.u < pi.expand(batch_size, self.dim_z)] = 1
                self.train_z = z
            else:
                z = pi.expand(batch_size, self.dim_z)
                z[z < .5] = 0
                self.test_z = z
        else:
            pi2 = 1 - pi
            if self.u is None:
                raise Exception('Forward pass first')
            z = torch.FloatTensor(batch_size, self.dim_z).zero_().to(self.device)
            z[self.u > pi2.expand(batch_size, self.dim_z)] = 1

        return z.view(batch_size, self.dim_z, 1, 1)

    def forward(self, input_):
        """ forward for fc """
        if self.input_shape is None:
            self.input_shape = input_.size()
        b = None if not self.use_bias else self.bias
        output = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation)
        z = self.sample_z(output.size(0))
        output = output.mul(z)
        return output

    def activated_neurons(self):
        return (self.test_z > 0).sum()/self.test_z.size(0)


