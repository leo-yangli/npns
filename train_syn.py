import os
import sys
import time
import argparse

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import torch
import torch.nn.functional as nfunc
from torch.utils.data import TensorDataset, DataLoader
from six.moves import cPickle

from torch.utils.tensorboard import SummaryWriter
from models.NPN_MLP import NPNMLP

parser = argparse.ArgumentParser(description='PyTorch NPNs Synthetic Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

parser.add_argument('--batch_size', default=32, type=int, help='batch size')

# parser.add_argument('--k', default=7, type=float, help='k')
# parser.add_argument('--mode', default='sparse', type=str, help='training mode: expand or sparse')
# parser.add_argument('--init_size', default=(100, 80), type=int, nargs='+',help='initial size of the MLP (1,2 layer)')
# parser.add_argument('--stage1', default=500, type=int, help='the end epoch of stage 1')
# parser.add_argument('--stage2', default=1000, type=int, help='the end epoch of stage 2')
# parser.add_argument('--lambas', default=(0.35, 14), nargs='+', type=float, help='l0 regularization strength')

parser.add_argument('--k', default=1, type=float, help='k')
parser.add_argument('--mode', default='expand', type=str, help='training mode: expand or sparse')
parser.add_argument('--init_size', default=(3, 3), type=int, nargs='+', help='initial size of the MLP (1,2 layer)')
parser.add_argument('--stage1', default=100, type=int, help='the end epoch of stage 1')
parser.add_argument('--stage2', default=1000, type=int, help='the end epoch of stage 2')
parser.add_argument('--lambas', default=(.35, 14), type=float, nargs='+', help='l0 regularization strength')

parser.add_argument('--print_freq', default=200, type=int, help='print freq')
parser.add_argument('--dataset', type=str, default='1', help='syndata-1, syndata-2 (default: 1)')
parser.add_argument('--data-dir', type=str, default='./data', help='default: ./dataset')
parser.add_argument('--iterations', type=int, default=1000, metavar='N', help='number of iterations (default: 1000)')
parser.add_argument('--times', type=int, default=2, metavar='N', help='N experiments (default: 1)')
parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')

parser.add_argument('--moment', type=float, default=0.1, help='momentum for SGD (default: 0.1)')
parser.add_argument('--eps', type=float, default=1, help='eps * perturbation (default: 1)')
parser.add_argument('--alpha', type=int, default=4, help='alpha for upper/lower bound, (default: 4)')
parser.add_argument('--dist-norm', type=str, default="inf", help='distance norm{inf, 1, 2}, (default: inf)')
parser.add_argument('--top-k', type=int, default=1, help='top k, 1-9, (default: 1)')
parser.add_argument('--use-h', action='store_true', default=False, help='use output h of hidden layers')
parser.add_argument('--use-logits', action='store_true', default=False, help='use logits, default use prob vectors')
parser.add_argument('--ul-logits', action='store_true', default=False, help='use logits for unlabel data, default use prob vectors')
parser.add_argument('--use-unit', action='store_true', default=False, help='use logits default use prob vectors')
parser.add_argument('--drop', type=float, default=0.5, help='dropout rate, (default: 0.5)')
parser.add_argument('--no-app', action='store_true', default=False, help='not use approximation')
parser.add_argument('--l2-wt', type=float, default=0, help='l2 weight decay, (default: 0)')
parser.add_argument('--threshold', type=float, default=1, help='threshold for margin points selection(default: 1)')
parser.add_argument('--log-dir', type=str, default='', metavar='S', help='tensorboard directory, (default: an absolute path)')
parser.add_argument('--vis', action='store_true', default=False, help='visual by tensor board')

args = parser.parse_args()
args.dir_path = "."

# use some parameters, pid and running time to mark the process
args_dict = vars(args)
run_time = time.strftime('%d%H%M', time.localtime(time.time()))
base_dir = os.path.join(os.environ['HOME'], 'project/runs') if not args.log_dir else args.log_dir

print("args in this experiment:\n%s" % '\n'.join(str(e) for e in sorted(vars(args).items())))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ', device)

# define loss function
def criterion(output, target_var):
    loss = torch.nn.CrossEntropyLoss().to(device)(output, target_var)
    reg_loss = model.regularization()
    total_loss = (loss + reg_loss).to(device)
    return total_loss


def visualize_contour(epoch, acc, model, d_i, x_data, y_data, basis, val_loader, args, with_lds=False, save_filename='prob_cont'):
    line_width = 10

    range_x = np.arange(-2.0, 2.1, 0.05)
    a_inv = linalg.inv(np.dot(basis, basis.T))
    train_x_org = np.dot(x_data, np.dot(basis.T, a_inv))
    test_x_org = np.zeros((range_x.shape[0] ** 2, 2))
    train_x_1_ind = np.where(y_data == 1)[0]
    train_x_0_ind = np.where(y_data == 0)[0]

    for t in range(range_x.shape[0]):
        for j in range(range_x.shape[0]):
            test_x_org[range_x.shape[0] * t + j, 0] = range_x[t]
            test_x_org[range_x.shape[0] * t + j, 1] = range_x[j]

    test_x = np.dot(test_x_org, basis)
    model.eval()
    f_p_y_given_x = model(torch.FloatTensor(test_x).to(device))
    pred = nfunc.softmax(f_p_y_given_x, dim=1)[:, 1].cpu().detach().numpy()

    z = np.zeros((range_x.shape[0], range_x.shape[0]))
    for t in range(range_x.shape[0]):
        for j in range(range_x.shape[0]):
            z[t, j] = pred[range_x.shape[0] * t + j]

    y, x = np.meshgrid(range_x, range_x)

    font_size = 15
    rc = 'r'
    bc = 'b'

    if d_i == "1":
        rescale = 1.0  # /np.sqrt(500)
        arc1 = Arc(xy=(0.5 * rescale, -0.25 * rescale), width=2.0 * rescale, height=2.0 * rescale, angle=0, theta1=270,
                   theta2=180, linewidth=line_width, alpha=0.15, color=rc)
        arc2 = Arc(xy=(-0.5 * rescale, +0.25 * rescale), width=2.0 * rescale, height=2.0 * rescale, angle=0, theta1=90,
                   theta2=360, linewidth=line_width, alpha=0.15, color=bc)
        fig = plt.gcf()
        fig.gca().add_artist(arc1)
        fig.gca().add_artist(arc2)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
    else:
        rescale = 1.0  # /np.sqrt(500)
        circle1 = Circle((0, 0), 1.0 * rescale, color=rc, alpha=0.2, fill=False, linewidth=line_width)
        circle2 = Circle((0, 0), 0.15 * rescale, color=bc, alpha=0.2, fill=False, linewidth=line_width)
        fig = plt.gcf()
        fig.gca().add_artist(circle1)
        fig.gca().add_artist(circle2)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

    try:
        levels = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
        cs = plt.contour(x * rescale, y * rescale, z, 7, cmap='bwr', vmin=0., vmax=1.0, linewidths=8., levels=levels)
        cbar = plt.colorbar(cs)
        cbar.ax.tick_params(labelsize=font_size)
        plt.setp(cs.collections, linewidth=1.0)
        plt.contour(x * rescale, y * rescale, z, 1, cmap='binary', vmin=0, vmax=0.5, linewidths=2.0)

        plt.xlim([-2. * rescale, 2. * rescale])
        plt.ylim([-2. * rescale, 2. * rescale])
        plt.xticks([-2.0, -1.0, 0, 1, 2.0], fontsize=font_size)
        plt.yticks([-2.0, -1.0, 0, 1, 2.0], fontsize=font_size)

        plt.scatter(train_x_org[train_x_1_ind, 0] * rescale, train_x_org[train_x_1_ind, 1] * rescale, s=10, marker='o', c=rc, label='$y=1$')
        plt.scatter(train_x_org[train_x_0_ind, 0] * rescale, train_x_org[train_x_0_ind, 1] * rescale, s=10, marker='^', c=bc, label='$y=0$')
        plt.title('#{}, Accuracy: {:.2f}%, # of parameters: {}'.format(epoch, acc*100, model.para_num()), fontsize=font_size)

        plt.savefig(save_filename + '.png')
        plt.savefig(save_filename + '.pdf')
        # plt.show(block=False)
        plt.close()
    except():
        print("error")


from time import localtime

current_time = time.strftime('%Y-%m-%d %H:%M:%S', localtime())
with open('%s/syndata_%s.pkl' % (args.data_dir, args.dataset), "rb") as f:
    if sys.version_info.major == 3:
        dataset = cPickle.load(f, encoding='bytes')
    else:
        dataset = cPickle.load(f)

x_train = torch.FloatTensor(np.asarray(dataset[0][1][0][0:500]))
t_train = torch.LongTensor(np.asarray(dataset[0][1][1][0:500]))
x_valid = torch.FloatTensor(np.asarray(dataset[0][1][0][500:]))
t_valid = torch.LongTensor(np.asarray(dataset[0][1][1][500:]))

train_loader = DataLoader(TensorDataset(x_train, t_train), 128, False)
val_loader = DataLoader(TensorDataset(x_valid, t_valid), 128, False)

error_rate = 1.0
best_err_rate = 100
best_model = None
avg_error = 0
avg_local_error = 0

local_best_rate = 100
args.dist_norm = int(args.dist_norm) if args.dist_norm != "inf" else np.inf


# tensorboard
writer = SummaryWriter(f'logs/{args.mode}/{time.localtime()}')
# model
model = NPNMLP(init_size=args.init_size, lambas=args.lambas, device=device).to(device)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), args.lr)
# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
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
    test_name = "plot/{}/test{}".format(args.mode, str(epoch).zfill(4))
    if not os.path.isdir("plot/{}".format(args.mode)):
        os.makedirs("plot/{}".format(args.mode))
    visualize_contour(epoch, correct / total, model, args.dataset, x_valid, t_valid, dataset[1], val_loader, args,
                      with_lds=True, save_filename=test_name)

    return val_loss / total

prev_loss = 100
for epoch in range(1, 2000):
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
                model.expand(0.1)
                prev_loss = val_loss


