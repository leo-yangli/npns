import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import random
from models import NPNLeNet5
import argparse
import time

parser = argparse.ArgumentParser(description='PyTorch NPNs MNIST Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

parser.add_argument('--batch_size', default=128, type=int, help='batch size')

# parser.add_argument('--k', default=7, type=float, help='k')
# parser.add_argument('--mode', default='sparse', type=str, help='training mode: expand or sparse')
# parser.add_argument('--init_size', default=(20, 50, 500), type=int, nargs='+', help='initial size of the LeNet (1,2,3 layer)')

parser.add_argument('--k', default=1, type=float, help='k')
parser.add_argument('--mode', default='expand', type=str, help='training mode: expand or sparse')
parser.add_argument('--init_size', default=(3, 3, 3), type=int, nargs='+', help='initial size of the LeNet (1,2,3 layer)')

parser.add_argument('--lambas', default=(10, 0.5, 0.1, 10), type=int, nargs='+', help='l0 regularization strength')
parser.add_argument('--stage1', default=100, type=int, help='the end epoch of stage 1')
parser.add_argument('--stage2', default=350, type=int, help='the end epoch of stage 2')
parser.add_argument('--print_freq', default=200, type=int, help='print freq')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# set seed
torch.backends.cudnn.benchmark = True
setup_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ', device)

# load dataset
transform_data = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                               transform=transform_data),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, transform=transform_data),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# tensorboard
writer = SummaryWriter(f'logs/{args.mode}/{time.localtime()}')
# model
model = NPNLeNet5(init_size=args.init_size, lambas=args.lambas, device=device).to(device)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), args.lr)
# scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], gamma=0.1)
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
        optimizer.zero_grad()
        outputs = model(input_, target)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

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
        print('[Epoch {}] Validation Loss: {:.2f} | Accuracy: {:.2f}% | # of Paramters: {}'.format(epoch,
               val_loss / (batch_idx + 1), 100. * correct / total, model.para_num()))
        for (i, num) in enumerate(model.get_activated_neurons()):
            writer.add_scalar("val_layer/{}".format(i), num, global_step=epoch)
        writer.add_scalar('val/loss', val_loss / (batch_idx + 1), global_step=epoch)
        writer.add_scalar('val/accuracy', correct / total, global_step=epoch)
        writer.add_scalar("# of parameters", model.para_num(), global_step=epoch)
    return val_loss / total


def main():
    prev_loss = 100
    for epoch in range(500):
        if epoch == 0:
            print('Stage 1')
            model.set_k(5000)

        if epoch == args.stage1:
            print('Stage 2')
            if args.mode == 'expand':
                model.set_k(args.k)
            else:
                model.set_k(args.k)

        if epoch == args.stage2:
            print('Stage 3')
            model.set_k(5000)
        train(epoch)
        val_loss = val(epoch)
        scheduler.step(epoch)
        writer.add_scalar('k', model.k, global_step=epoch)
        if args.stage1 < epoch < args.stage2:
            if args.mode == 'expand':
                if val_loss < prev_loss:
                    model.expand(0)
                    prev_loss = val_loss

main()
