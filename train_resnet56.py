import argparse
import random
import time

import numpy as np
import torch
import torch.optim as optim
from resnet56_models.NPN_ResNet56 import resnet56
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloaders import cifar10, cifar100

parser = argparse.ArgumentParser(description='PyTorch NPNs Training')
parser.add_argument('--lr_resnet', default=0.1, type=float, help='learning rate of resnet')
parser.add_argument('--lr_gate', default=0.001, type=float, help='learning rate of gate')
parser.add_argument('---momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=220, type=int, help='training epochs')
parser.add_argument('--k', default=7, type=float, help='k')
parser.add_argument('--mode', default='expand', type=str, help='training mode: expand or sparse')
parser.add_argument('--init_factor', default=-1, type=float, nargs='+',  help='initial size of the ResNet 56')
parser.add_argument('--lamba', default=1e-5, type=float, help='l0 regularization strength')
parser.add_argument('--stage1', default=20, type=int, help='the end epoch of stage 1')
parser.add_argument('--stage2', default=200, type=int, help='the end epoch of stage 2')
parser.add_argument('--print_freq', default=20, type=int, help='print freq')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ', device)

# load dataset

dataload = cifar10 if args.num_classes == 10 else cifar100
train_loader, val_loader, num_classes = dataload(augment=True, batch_size=args.batch_size)

# tensorboard
writer = SummaryWriter(f'logs/{args.mode}/{time.localtime()}')
# model
model = resnet56(args).to(device)
# optimizer

resnet_params = []
gate_params = []
for name, param in model.named_parameters():
    if 'z_phi' in name:
        gate_params.append(param)
    else:
        resnet_params.append(param)

MILESTONES = [60, 120, 180, 240, 300]

optimizer_resnet = optim.SGD(resnet_params, lr=args.lr_resnet, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler_resnet = optim.lr_scheduler.MultiStepLR(optimizer_resnet, milestones=MILESTONES, gamma=args.lr_decay)
optimizer_gate = optim.Adam(gate_params, lr=args.lr_gate)
if args.mode == 'sparse':
    scheduler_gate = optim.lr_scheduler.MultiStepLR(optimizer_gate, milestones=MILESTONES[1:], gamma=args.lr_decay)

# loss function
def criterion(output, target):
    loss = torch.nn.CrossEntropyLoss().to(device)(output, target)
    total_loss = (model.regularization() + loss).to(device)
    return total_loss


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (input_, target) in enumerate(train_loader):
        input_, target = input_.to(device), target.to(device)
        model.zero_grad()
        outputs = model(input_, target)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer_resnet.step()
        if args.stage1 < epoch < args.stage2:
            optimizer_gate.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        if batch_idx % args.print_freq == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', train_loss / (batch_idx + 1), global_step=step)
            writer.add_scalar('train/accuracy', correct / total, global_step=step)
            print('[Epoch {}] Train Loss: {:.2f} | Accuracy: {:.2f}%'.format(epoch,
                    train_loss / (batch_idx + 1), 100. * correct / total))


def val(epoch):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (input_, target) in enumerate(val_loader):
            input_, target = input_.to(device), target.to(device)
            output = model(input_)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        print('[Epoch {}] Validation Loss: {:.2f} | Accuracy: {:.2f}% '.format(epoch,
               val_loss / (batch_idx + 1), 100. * correct / total))
        for (i, num) in enumerate(model.get_activated_neurons()):
            writer.add_scalar("val_layer/{}".format(i), num, global_step=epoch)
        writer.add_scalar('val/loss', val_loss / (batch_idx + 1), global_step=epoch)
        writer.add_scalar('val/accuracy', correct / total, global_step=epoch)
        for (i, gate) in enumerate(gate_params):
            writer.add_histogram("layer/{}".format(i), gate, global_step=epoch)
    return val_loss / total


def main():
    prev_loss = 100
    for epoch in range(args.epochs):
        if epoch == 0:
            print('Stage 1, k = 5000')
            model.set_k(5000)

        if epoch == args.stage1:
            print('Stage 2, k =', args.k)
            model.set_k(args.k)

        if epoch == args.stage2:
            print('Stage 3, k = 5000')
            model.set_k(5000)

        train(epoch)
        val_loss = val(epoch)
        if args.mode == 'sparse':
            scheduler_gate.step(epoch)
        scheduler_resnet.step(epoch)
        writer.add_scalar('k', model.k, global_step=epoch)
        if args.stage1 < epoch < args.stage2:
            if args.mode == 'expand':
                if val_loss < prev_loss:
                    model.expand(3)
                    prev_loss = val_loss

main()
