import os
import argparse
import numpy as np
import torch
from torch import nn, optim

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image
from torchvision import datasets, transforms


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        # self.fc1 = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 1)
        # self.fc3 = nn.Linear(10, 1000)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64 * 28 * 28)
        # y_ = self.fc3(labels)
        # y_ = F.relu(y_)
        # x = torch.cat([x, y_], 1)  # add label 1000 dimension
        x = self.fc1(x)
        x = F.relu(x)
        rf = self.fc2(x)
        c = self.fc3(x)
        return torch.sigmoid(rf.view(-1)), c


class ModelG(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelG, self).__init__()
        self.fc2 = nn.Linear(10, 1000)
        self.fc = nn.Linear(self.z_dim + 1000, 64 * 28 * 28)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc2(labels)  # 1000 dimensions
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)  # 100 dimensions code
        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DCGAN')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default=128)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default=0.01)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs.')
    parser.add_argument('--nz', type=int, default=100,
                        help='Number of dimensions for input noise.')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Enable cuda')
    parser.add_argument('--save_every', type=int, default=1,
                        help='After how many epochs to save the model.')
    parser.add_argument('--print_every', type=int, default=50,
                        help='After how many epochs to print loss and save output samples.')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Path to save the trained models.')
    parser.add_argument('--samples_dir', type=str, default='samples',
                        help='Path to save the output samples.')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.samples_dir):
        os.mkdir(args.samples_dir)

    INPUT_SIZE = 784
    SAMPLE_SIZE = 80
    NUM_LABELS = 10
    train_dataset = datasets.MNIST(root='../mnist',
                                   train=True,
                                   download=False,
                                   transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=args.batch_size)

    model_d = ModelD()
    model_g = ModelG(args.nz)
    criterion = nn.BCELoss()
    criterion_class = nn.CrossEntropyLoss()

    fixed_noise = torch.FloatTensor(SAMPLE_SIZE, args.nz).normal_(0, 1)
    fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS)
    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i * (SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0

    if args.cuda:
        model_d.cuda()
        model_g.cuda()
        fixed_noise = fixed_noise.cuda()
        fixed_labels = fixed_labels.cuda()

    optim_d = optim.SGD(model_d.parameters(), lr=args.lr)
    optim_g = optim.SGD(model_g.parameters(), lr=args.lr)
    fixed_noise = Variable(fixed_noise)
    fixed_labels = Variable(fixed_labels)

    for epoch_idx in range(args.epochs):
        model_d.train()
        model_g.train()

        d_loss = 0.0
        g_loss = 0.0
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            batch_size = train_x.size(0)
            train_x = train_x.cuda()
            train_y = train_y.cuda()

            real_label = torch.ones(batch_size).cuda()
            fake_label = torch.zeros(batch_size).cuda()

            one_hot_labels = torch.zeros(batch_size, NUM_LABELS).cuda()
            one_hot_labels.scatter_(1, train_y.view(batch_size, 1), 1)

            output, class_ = model_d(train_x)
            optim_d.zero_grad()
            errD_real = criterion(output, real_label)
            errD_real_class = criterion_class(class_, train_y)
            errD_real.backward(retain_graph=True)
            errD_real_class.backward()
            realD_mean = output.detach().cpu().mean()

            one_hot_labels.zero_()
            shuffle = (torch.randint(0, NUM_LABELS-1, (batch_size,)) + 1).type(torch.cuda.LongTensor)
            rand_y = (train_y + shuffle) % NUM_LABELS
            one_hot_labels.scatter_(1, rand_y.view(batch_size, 1), 1)
            noise = torch.FloatTensor(batch_size, args.nz).normal_(0, 1).cuda()

            g_out = model_g(noise, one_hot_labels)
            output, class_ = model_d(g_out.detach())
            errD_fake = criterion(output, fake_label)
            errD_fake_class = criterion_class(class_, rand_y)
            fakeD_mean = output.data.cpu().mean()
            errD_fake.backward(retain_graph=True)
            errD_fake_class.backward()
            errD = errD_real+errD_fake
            optim_d.step()

            # train the G
            output, class_ = model_d(g_out)
            errG = criterion(output, real_label)
            errG_class = criterion_class(class_, rand_y)
            optim_g.zero_grad()
            errG.backward(retain_graph=True)
            errG_class.backward()
            optim_g.step()

            d_loss += errD.data.item()
            g_loss += errG.data.item()
            if batch_idx % args.print_every == 0:
                print(
                    "\t{} ({} / {}) mean D(fake) = {:.4f}, mean D(real) = {:.4f}".
                        format(epoch_idx, batch_idx, len(train_loader), fakeD_mean,
                               realD_mean))

                g_out = model_g(fixed_noise, fixed_labels).data.view(
                    SAMPLE_SIZE, 1, 28, 28).cpu()
                save_image(g_out,
                           '{}/{}_{}.png'.format(
                               args.samples_dir, epoch_idx, batch_idx))

        print('Epoch {} - D loss = {:.4f}, G loss = {:.4f}'.format(epoch_idx,
                                                                   d_loss, g_loss))
        if epoch_idx % args.save_every == 0:
            torch.save({'state_dict': model_d.state_dict()},
                       '{}/model_d_epoch_{}.pth'.format(
                           args.save_dir, epoch_idx))
            torch.save({'state_dict': model_g.state_dict()},
                       '{}/model_g_epoch_{}.pth'.format(
                           args.save_dir, epoch_idx))
