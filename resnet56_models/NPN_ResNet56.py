"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from resnet56_models.layer import ARMConv2dBn, ARMDense
import math


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, name='', k=7, lamba=1e-5, init_factor=.2, device='cpu'):
        super().__init__()
        self.conv1 = ARMConv2dBn(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, droprate_init=0.5,
                                 k=k, lamba=lamba, local_rep=True, name=name+'.1',
                                 init_size=int(out_channels*init_factor), device=device)

        self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = ARMConv2dBn(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False,
        #                          droprate_init=0.5, k=k, lamba=lamba, local_rep=True, name=name+'.2',
        #                          init_size=int(out_channels * BasicBlock.expansion * init_factor), device=device)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1,
                                             bias=False,), nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        # shortcut
        self.shortcut = nn.Sequential()
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            # self.shortcut = ARMConv2dBn(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
            #                             bias=False, init_size=-1,
            #                             name=name+'.shortcut', device=device)
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
                                        bias=False,), nn.BatchNorm2d(out_channels * BasicBlock.expansion))


    def forward(self, x):
        conv = self.conv1(x)
        conv = self.relu1(conv)
        conv = self.conv2(conv)
        sc = self.shortcut(x)
        return nn.ReLU(inplace=True)(conv + sc)


class BlocksContainer(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_blocks, stride, name=None, init_factor=.2,
                   lamba=0.01, k=7., device='cpu'):
        super().__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        self.fine_tune = False
        self.blocks = nn.Sequential()

        for i, stride in enumerate(strides):
            self.blocks.add_module('block%d' % i,
                    block(in_channels, out_channels, stride, name='{}.{}'.format(name, i), init_factor=init_factor,
                     k=k, lamba=lamba, device=device))
            in_channels = out_channels * block.expansion

    def flops_params(self):
        f_p = torch.zeros(2)
        for l in self.blocks:
            f_p += l.flops_params()
        return f_p


    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x



# ResNet56 for CIFAR dataset
class ResNet56(nn.Module):

    def __init__(self, block, num_block, num_classes=100, lamba=0.01, k=7, init_factor=(-1,), device='cpu'):
        super().__init__()
        if len(init_factor) == 1:
            init_factor = init_factor * 5
        self.threshold = None
        self.in_channels = 16
        self.device = device
        self.ar = False
        self.conv1 = ARMConv2dBn(3, 16, kernel_size=3, stride=1, padding=1, bias=False, droprate_init=0.5,
                    k=k, lamba=lamba, local_rep=True, name='conv1',
                    init_size=int(16 * init_factor[0]), device=device)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2_x = BlocksContainer(block, 16, 16, num_block[0], 1, name='conv2', init_factor=init_factor[1],
                                       lamba=lamba, k=k, device=device)
        self.conv3_x = BlocksContainer(block, 16, 32, num_block[1], 2, name='conv3', init_factor=init_factor[2],
                                       lamba=lamba, k=k, device=device)
        self.conv4_x = BlocksContainer(block, 32, 64, num_block[2], 2, name='conv4', init_factor=init_factor[3],
                                       lamba=lamba, k=k, device=device)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ARMDense(64, num_classes, name='fc', init_size=int(64*init_factor[4]), lamba=lamba, k=k,  device=device)
        self.conv_layers, self.fc_layers, self.bn_layers, self.bn_params = [], [], [], []
        for m in self.modules():
            if isinstance(m, ARMConv2dBn):
                self.conv_layers.append(m)
            elif isinstance(m, ARMDense):
                self.fc_layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]
                self.bn_layers.append(m)
        self.layers = [*self.conv_layers, *self.fc_layers]

    def set_k(self, k):
        self.k = k
        for l in self.conv_layers:
            l.k = k
        for l in self.fc_layers:
            l.k = k


    def score(self, x):
        output = self.relu1(self.conv1(x))
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def forward(self, x, y=None):
        if self.training:
            self.forward_mode(True)
            score = self.score(x)

            self.eval()
            if self.ar is not True:
                self.forward_mode(False)
                score2 = self.score(x).data
                f1 = nn.CrossEntropyLoss()(score2, y).data
            else:
                f1 = 0
            f2 = nn.CrossEntropyLoss()(score, y).data

            self.update_phi_gradient(f1, f2)
            self.train()
        else:
            self.forward_mode(True)
            score = self.score(x)
        return score

    def forward_mode(self, mode):
        for layer in self.layers:
            layer.forward_mode = mode


    def update_phi_gradient(self, f1, f2):
        for layer in self.layers:
            layer.update_phi_gradient(f1, f2)

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            reg = layer.regularization()
            regularization += reg

        # regularization = regularization.cuda()
        return regularization

    def get_activated_neurons(self):
        return [layer.activated_neurons() for layer in self.layers]

    def z_phis(self):
        return [layer.z_phi for layer in self.layers]

    def expand(self, fill=0):
        for l in self.layers:
            l.expand_layer(fill)

def resnet56(args):
    return ResNet56(BasicBlock, [9, 9, 9], num_classes=args.num_classes,  init_factor=args.init_factor,
                 lamba=args.lamba, device='cuda')
