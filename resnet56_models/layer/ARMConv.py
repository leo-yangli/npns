import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair as pair
import torch.nn.functional as F
import math

class ARMConv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, name='',
                 weight_decay=0, lamba=0.1/6e5, droprate_init=0.01, k=7, local_rep=True, init_size=-1, device='cpu'):
        super(ARMConv2dBn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = int(out_channels)
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
            self.bias = Parameter(torch.Tensor(self.out_channels))
        self.weights = Parameter(torch.Tensor(self.out_channels, in_channels, *self.kernel_size))
        self.z_phi = Parameter(torch.Tensor(self.out_channels))
        self.dim_z = self.out_channels
        self.input_shape = None
        self.u = torch.Tensor(self.dim_z).uniform_(0, 1)
        self.droprate_init = droprate_init
        self.forward_mode = True
        self.local_rep = local_rep
        self.activated_neuron_size = init_size
        self.device = device
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.reset_parameters()
        self.layer_name = name
        self.dimz_tensor = torch.FloatTensor(self.dim_z).zero_().to(device)
        print(self)

    def __repr__(self):
        s = ('{name}-{layer_name}-({in_channels}, {out_channels}, kernel_size={kernel_size} '
             ', stride={stride}, weight_decay={weight_decay}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weights, mode='fan_out')
        if self.activated_neuron_size <= 0:
            # self.z_phi.data.fill_(114.5/self.k)
            self.z_phi.data.normal_(3/7, 1e-2)
        else:
            print('{}:{}'.format(self.activated_neuron_size, self.dim_z))
            self.z_phi.data[:self.activated_neuron_size].normal_(3 / self.k, 1e-2 / self.k)
            self.z_phi.data[self.activated_neuron_size:].normal_(-40 / self.k, 1e-2 / self.k)
        if self.use_bias:
            self.bias.data.fill_(0)

    def expand_layer(self, fill):
        add_num = 1 #  math.ceil(min(self.out_channels * 0.05, self.out_channels - self.activated_neuron_size))
        if self.activated_neuron_size + add_num < self.out_channels and self.activated_neurons() >= self.activated_neuron_size:
            self.z_phi.data[self.activated_neuron_size:self.activated_neuron_size+add_num].normal_(fill / self.k, 1e-2)
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
            z = self.dimz_tensor.clone().expand(batch_size, self.dim_z).zero_()
            if self.training:
                if self.local_rep:
                    self.u = self.dimz_tensor.clone().uniform_(0, 1).expand(batch_size, self.dim_z)
                else:
                    self.u = self.dimz_tensor.clone().expand(batch_size, self.dim_z).uniform_(0, 1)

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
            z = self.dimz_tensor.clone().expand(batch_size, self.dim_z)
            z[self.u > pi2.expand(batch_size, self.dim_z)] = 1

        return z.view(batch_size, self.dim_z, 1, 1)

    def forward(self, input_):
        """ forward for fc """
        if self.input_shape is None:
            self.input_shape = input_.size()
        b = None if not self.use_bias else self.bias
        output = F.conv2d(input_, self.weights, b, self.stride, self.padding, self.dilation)
        z = self.sample_z(output.size(0))
        output = self.bn(output)
        output = output.mul(z)
        return output

    def activated_neurons(self):
        return (self.test_z > 0).sum()/self.test_z.size(0)


