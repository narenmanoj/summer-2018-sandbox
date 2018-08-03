import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.autograd import Function

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = torch.cuda.is_available()

print("Cuda: " + str(cuda))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def dc_weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# taken from https://github.com/maitek/waae-pytorch/blob/master/WAAE.py
class Decoder(nn.Module):
    def __init__(self, latent_dim=64, channels=3, ngf=64, ndf=64):
        super(Decoder, self).__init__()
        self.main = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf, channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim=64, channels=3, ngf=64, ndf=64):
        super(Encoder, self).__init__()
        nc = channels
        nz = latent_dim
        self.main = [
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=False),
        ]

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, latent_dim=64, channels=3, ngf=64, ndf=64, h_dim=128):
        super(Discriminator, self).__init__()
        nz = latent_dim
        nc = channels
        self.main = [
            nn.Linear(nz, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)
    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


def show(img):
    npimg = img.clone().cpu().numpy()
    num_images = npimg.shape[0]
    nrows = int(np.ceil(np.sqrt(num_images)))
    ncols = nrows
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(
                np.transpose(npimg[i], (1, 2, 0)), interpolation='nearest')
            i += 1
            if i >= num_images:
                return


def get_mnist_dataloader(batch_size, img_size=64, train=True):
    assert batch_size % 3 == 0  # for stacking
    assert 60000 % batch_size == 0  # to account for packing
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


def pack_samples(imgs, packing=1):
    assert packing >= 1
    if packing != 1:
        assert len(imgs) % packing == 0
    batch = []
    for i in range(0, len(imgs), packing):
        sub_batch = []
        for j in range(packing):
            sub_batch.append(imgs[i + j])
        batch.append(torch.cat(sub_batch))
    return torch.stack(batch)


def form_stacks(imgs):
    assert len(imgs) % 3 == 0  # for stacking
    batch = []
    for i in range(0, len(imgs), 3):
        # stack imgs[i], imgs[i + 1], imgs[i + 2] across RGB channels
        new_img = torch.cat([imgs[i], imgs[i + 1], imgs[i + 2]])
        batch.append(new_img)
    return torch.stack(batch)


def train(save_model=False,
          filename="",
          num_epochs=10000,
          num_samples_per_batch=100,
          latent_dim=64,
          disp_interval=10,
          lr=1e-4):
    # num_samples is for plotting purposes
    if save_model:
        assert len(filename) > 0

    Q = Encoder()
    P = Decoder()
    D = Discriminator()
    
    def reset_grad():
        Q.zero_grad()
        P.zero_grad()
        D.zero_grad()
        
    Q_solver = torch.optim.Adam(Q.parameters(), lr=lr, amsgrad=True)
    P_solver = torch.optim.Adam(P.parameters(), lr=lr, amsgrad=True)
    D_solver = torch.optim.Adam(D.parameters(), lr=lr*0.1, amsgrad=True)
    
    if cuda:
        Q.cuda()
        P.cuda()
        D.cuda()
    
    dataloader = get_mnist_dataloader(3 * num_samples_per_batch)
    
    for epoch in range(num_epochs):
        print(epoch)
        for i, (imgs, _) in enumerate(dataloader):
            X = form_stacks(Variable(imgs.type(Tensor), requires_grad=False))
            z_sample = Q(X)
            X_sample = P(z_sample)
            recon_loss = F.mse_loss(X_sample, X)
            
            recon_loss.backward()
            P_solver.step()
            Q_solver.step()
            reset_grad()
            
            for __ in range(5):
                z_real = Variable(Tensor(
                    np.random.normal(loc=0.0, scale=1.0, size=(X.shape[0], latent_dim))), requires_grad=False)
                z_fake = Q(X).view(X.shape[0], -1)
                D_real = D(z_real)
                D_fake = D(z_fake)
                
                D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

                D_loss.backward()
                D_solver.step()
                
                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01)
                
                reset_grad()
                
            z_fake = Q(X).view(X.shape[0], -1)
            D_fake = D(z_fake)

            G_loss = -torch.mean(D_fake)
            
            G_loss.backward()
            Q_solver.step()

            reset_grad()
            
        if epoch % disp_interval == 0:
            print("[Epoch %d/%d] [Discriminator Loss: %f] [Generator Loss: %f] [Reconstruction Loss: %f]"
                  % (epoch, num_epochs, D_loss.item(), G_loss.item(), recon_loss.item()))
            num_samples_to_test = 10
            z = Variable(
                Tensor(
                    np.random.normal(0, 1, (num_samples_to_test, latent_dim, 1, 1))))
            samples = P(z)
            show(samples.detach())
            plt.show()
    
    if save_model:
        torch.save(Q.state_dict(), filename + "_encoder")
        torch.save(P.state_dict(), filename + "_decoder")
        torch.save(D.state_dict(), filename + "_discriminator")
    return Q, P, D


def load_model(filename, latent_dim=100, img_size=32):
    generator = DCGenerator(latent_dim=latent_dim, img_size=img_size)
    generator.load_state_dict(torch.load(filename))
    if cuda:
        generator.cuda()
    return generator


def gradient_free_radius(z,
                         loaded_gen,
                         latent_dim=2,
                         grid_length=5,
                         num_trials=10):
    znp = z.cpu().detach().numpy()
    znp_gan_eval = loaded_gen(z).cpu().detach().numpy()
    z_class = point_to_index(znp_gan_eval, grid_length=grid_length)

    dist = 0.0
    best_global_point = None
    for i in range(num_trials):
        # sample a point z_prime at random with a different classification
        z_prime = Variable(Tensor(np.random.normal(0, 1, (1, latent_dim))))
        zpnp = z_prime.cpu().detach().numpy()
        zpnp_gan_eval = loaded_gen(z_prime).cpu().detach().numpy()
        z_prime_class = point_to_index(zpnp_gan_eval, grid_length=grid_length)
        while z_prime_class == z_class:
            z_prime = Variable(Tensor(np.random.normal(0, 1, (1, latent_dim))))
            zpnp = z_prime.cpu().detach().numpy()
            zpnp_gan_eval = loaded_gen(z_prime).cpu().detach().numpy()
            z_prime_class = point_to_index(
                zpnp_gan_eval, grid_length=grid_length)

        # print("[Z Class: %d] [Z prime Class: %d]" % (z_class, z_prime_class))
        # print((znp, zpnp))
        # compute largest lamb s.t. lamb * z + (1 - lamb) * z_prime
        # has a different classification index from z

        high, low = 1.0, 0.0
        lamb = 0.50
        equal_eps = 0.0000001
        while high > low + equal_eps:
            lamb = (high + low) / 2.0
            test_z = lamb * znp + (1 - lamb) * zpnp
            test_z_tensor = Variable(Tensor(test_z))
            test_z_eval = loaded_gen(test_z_tensor).cpu().detach().numpy()
            test_z_class = point_to_index(test_z_eval, grid_length=grid_length)
            if test_z_class == z_class:
                # this means that lamb is too high
                high = lamb
            else:
                # this means that lamb is too low
                low = lamb
        best_point = lamb * znp + (1 - lamb) * zpnp
        best_point_tensor = Variable(Tensor(best_point))
        best_point_eval = loaded_gen(best_point_tensor).cpu().detach().numpy()
        best_point_class = point_to_index(
            best_point_eval, grid_length=grid_length)

        # print((z_class, best_point_class))
        d = np.linalg.norm(best_point - z)
        if dist == 0:
            dist = d
            best_global_point = best_point.copy()
        dist = min(dist, d)
        if dist == d:
            best_global_point = best_point.copy()

    return dist, best_global_point


def gradient_free_avg_radius(loaded_gen,
                             latent_dim=2,
                             grid_length=5,
                             num_trials=1000):
    total_dist = 0.0
    for i in range(1, num_trials + 1):
        if i % (num_trials // 10) == 0:
            print("[Iteration %d] [Distance: %f]" % (i, total_dist / i))
        num_sub_trials = 20
        z = Variable(Tensor(np.random.normal(0, 1, (1, latent_dim))))
        dist, best_global_point = gradient_free_radius(
            z,
            loaded_gen,
            latent_dim=latent_dim,
            grid_length=grid_length,
            num_trials=num_sub_trials)
        total_dist += dist
    avg_dist = total_dist / num_trials
    return avg_dist