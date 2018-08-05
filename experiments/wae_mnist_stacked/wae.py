import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = torch.cuda.is_available()

print("Cuda: " + str(cuda))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# utility functions
def get_dataloader(dataset_name="MNIST",
                   batch_size=100,
                   img_size=28,
                   train=True):
    if dataset_name == "MNIST":
        assert 60000 % batch_size == 0
        os.makedirs("./data/mnist", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data/mnist",
                train=train,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])),
            batch_size=batch_size,
            shuffle=True)
        return dataloader


def show(img):
    npimg = img.clone().cpu().numpy()
    num_images = npimg.shape[0]
    nrows = int(np.ceil(np.sqrt(num_images)))
    ncols = nrows
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            plt.subplot(nrows, ncols, i + 1)
            to_show = np.clip(np.transpose(npimg[i], (1, 2, 0)), 0, 1)
            plt.imshow(to_show, interpolation='nearest')
            i += 1
            if i >= num_images:
                return


def set_zero_grad(networks):
    for n in networks:
        n.zero_grad()


def set_eval(networks):
    for n in networks:
        n.eval()


def set_train(networks):
    for n in networks:
        n.train()


def conv_block(in_filters,
               out_filters,
               conv_filter_size,
               stride=1,
               padding=1,
               bn=True,
               relu=True):
    block = [
        nn.Conv2d(
            in_filters,
            out_filters,
            conv_filter_size,
            stride=stride,
            padding=padding)
    ]
    if bn:
        block.append(nn.BatchNorm2d(out_filters))
    if relu:
        block.append(nn.ReLU())
    return block


def deconv_block(in_filters,
                 out_filters,
                 conv_filter_size,
                 stride=1,
                 padding=1,
                 bn=True,
                 relu=True):
    block = [
        nn.ConvTranspose2d(
            in_filters,
            out_filters,
            conv_filter_size,
            stride=stride,
            padding=padding)
    ]
    if bn:
        block.append(nn.BatchNorm2d(out_filters))
    if relu:
        block.append(nn.ReLU())
    return block


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


# models
class Encoder(nn.Module):
    def __init__(self,
                 img_size=28,
                 latent_dim=8,
                 conv_filter_size=4,
                 channels=1):
        super(Encoder, self).__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.conv_filter_size = conv_filter_size
        self.channels = channels
        self.stride = 2
        self.padding = (
            self.conv_filter_size - self.stride) // 2  # "SAME" padding

        # self.conv_model = nn.Sequential(nn.Conv2d(self.channels, 128, self.conv_filter_size, stride=self.stride, padding=self.padding),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(),
        #                            nn.Conv2d(128, 256, self.conv_filter_size, stride=self.stride, padding=self.padding),
        #                            nn.BatchNorm2d(256),
        #                            nn.ReLU(),
        #                            nn.Conv2d(256, 512, self.conv_filter_size, stride=self.stride, padding=self.padding),
        #                            nn.BatchNorm2d(512),
        #                            nn.ReLU(),
        #                            nn.Conv2d(512, 1024, self.conv_filter_size, stride=self.stride, padding=self.padding),
        #                            nn.BatchNorm2d(1024),
        #                            nn.ReLU())
        self.conv_model = nn.Sequential(
            *conv_block(
                self.channels,
                128,
                self.conv_filter_size,
                stride=self.stride,
                padding=self.padding),
            *conv_block(
                128,
                256,
                self.conv_filter_size,
                stride=self.stride,
                padding=self.padding),
            *conv_block(
                256,
                512,
                self.conv_filter_size,
                stride=self.stride,
                padding=self.padding),
            *conv_block(
                512,
                1024,
                self.conv_filter_size,
                stride=self.stride,
                padding=self.padding))

        # fix the dimensions here
        self.fc1 = nn.Linear(1024, self.latent_dim)

    def forward(self, img):
        out = self.conv_model(img)
        # do some reshaping thing here
        out = out.squeeze()
        return self.fc1(out)


class Decoder(nn.Module):
    def __init__(self,
                 img_size=28,
                 latent_dim=8,
                 conv_filter_size=4,
                 channels=1,
                 init_size=7):
        super(Decoder, self).__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.conv_filter_size = conv_filter_size
        self.channels = channels
        self.stride = 2
        self.padding = (
            self.conv_filter_size - self.stride) // 2  # "SAME" padding
        self.init_size = init_size

        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.init_size * self.init_size * 1024),
            nn.ReLU())

        # do some dimension fixing
        self.deconv_model = nn.Sequential(
            *deconv_block(
                self.init_size * self.init_size * 1024,
                512,
                self.conv_filter_size,
                stride=self.stride,
                padding=self.padding),
            *deconv_block(
                512,
                256,
                self.conv_filter_size,
                stride=self.stride,
                padding=self.padding),
            *deconv_block(
                256,
                self.channels,
                self.conv_filter_size,
                stride=self.stride,
                padding=self.padding,
                bn=False,
                relu=False), nn.Sigmoid())

    def forward(self, z):
        out = self.model(z)
        # do some reshaping thing here
        out = out.view(-1, self.latent_dim * 8, self.init_size, self.init_size)
        return self.deconv_model(out)


class Discriminator(nn.Module):
    def __init__(self, latent_dim=8):
        super(Discriminator, self).__init__()

        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 512), nn.ReLU(), nn.Linear(512, 512),
            nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512),
            nn.ReLU(), nn.Linear(512, 1))

    def forward(self, z):
        return self.model(z)


def train(lr=0.0001, epochs=100, latent_dim=8, sigma=1, lam=10):
    train_loader = get_dataloader()
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()
    if cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
    # Optimizers
    enc_optim = torch.optim.Adam(encoder.parameters(), lr=lr, amsgrad=True)
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=lr, amsgrad=True)
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr=0.5 * lr, amsgrad=True)

    enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
    dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
    dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)

    one = torch.Tensor([1])
    mone = one * -1

    if cuda:
        one = one.cuda()
        mone = mone.cuda()

    for epoch in range(epochs):
        step = 0

        for images, _ in tqdm(train_loader):

            if torch.cuda.is_available():
                images = images.cuda()

            set_zero_grad([encoder, decoder, discriminator])

            # ======== Train Discriminator ======== #

            frozen_params(decoder)
            frozen_params(encoder)
            free_params(discriminator)

            z_fake = torch.randn(images.size()[0], latent_dim) * sigma
            print("z_fake shape")
            print(z_fake.shape)

            if cuda:
                z_fake = z_fake.cuda()

            
            d_fake = discriminator(z_fake)
            print("d_fake shape")
            print(d_fake.shape)

            
            z_real = encoder(images)
            print("z_real shape")
            print(z_real.shape)
            
            d_real = discriminator(z_real)
            print("d_real shape")
            print(d_real.shape)
            
            log_d_fake = torch.log(d_fake).mean()
            
            log_d_fake.backward(mone)
            
            log_d_real = torch.log(1 - d_real).mean()
            log_d_real.backward(mone)

            dis_optim.step()

            # ======== Train Generator ======== #

            free_params(decoder)
            free_params(encoder)
            frozen_params(discriminator)

            batch_size = images.size()[0]

            z_real = encoder(images)
            x_recon = decoder(z_real)
            d_real = discriminator(encoder(Variable(images.data)))

            recon_loss = criterion(x_recon, images)
            d_loss = lam * (torch.log(d_real)).mean()

            recon_loss.backward(one)
            d_loss.backward(mone)

            enc_optim.step()
            dec_optim.step()

            step += 1
