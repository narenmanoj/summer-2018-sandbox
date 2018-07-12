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

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = torch.cuda.is_available()

print("Cuda: " + str(cuda))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# defining generator and discriminator architecture, taken from Github Pytorch-GAN (need to modify hyperparams)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(2,)):
        super(Generator, self).__init__()
        
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.0, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 400, normalize=True),
            *block(400, 400),
            *block(400, 400),
            *block(400, 400),
            nn.Linear(400, int(np.prod(img_shape)))
        )
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, img_shape=(2,)):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

def sample_from_2dgrid(grid_length=5, var=0.0025, num_samples=1):
    assert grid_length % 2 == 1
    samples = []
    # pick one of the Gaussians at random and then sample from it
    for i in range(num_samples):
        coords = np.random.randint(grid_length, size = 2)
        start = -2 * (grid_length // 2)
        mean = (coords[0] * 2 + start, coords[1] * 2 + start)
        p1 = np.random.normal(loc = mean[0], scale = np.sqrt(var))
        p2 = np.random.normal(loc = mean[1], scale = np.sqrt(var))
        samples.append((p1, p2))
    return samples

def train(save_model=False, filename="", num_samples=10000, num_epochs=10000, num_samples_per_batch=500, grid_length=5, var=0.0025, latent_dim=2):
    num_samples = 10000 # for plotting
    num_epochs = 10000
    num_samples_per_batch = 500
    grid_length = 5
    var = 0.0025
    latent_dim = 2
    # initialize things
    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator()
    adversarial_loss = torch.nn.BCELoss()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        
    # optimizers
    learning_rate = 0.0002
    b1 = 0.5
    b2 = 0.999
    amsgrad = True
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=8*learning_rate, betas=(b1, b2), amsgrad=amsgrad)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2), amsgrad=amsgrad)

    # underlay of true distribution
    real_samples = sample_from_2dgrid(grid_length=grid_length, num_samples=num_samples)
    x = [real_samples[i][0] for i in range(num_samples)]
    y = [real_samples[i][1] for i in range(num_samples)]

    # train the GAN
    for epoch in range(num_epochs):
        # ground truths
        valid = Variable(Tensor(num_samples_per_batch, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(num_samples_per_batch, 1).fill_(0.0), requires_grad=False)
        
        # input
        real_input = Variable(Tensor(sample_from_2dgrid(grid_length=grid_length, 
                                                        var=var, 
                                                        num_samples=num_samples_per_batch)))
        
        
        ########## Generator stuff ##########
        optimizer_G.zero_grad()
        # sample latent z
        z = Variable(Tensor(np.random.normal(loc=0.0, scale=1.0, size=(num_samples_per_batch, latent_dim))))
        
        # get generator output for the latent z
        fake_output = generator(z)
        
        # how well did we fool the discriminator?
        g_loss = adversarial_loss(discriminator(fake_output), valid)
        
        # gradient descent
        g_loss.backward()
        optimizer_G.step()
        
        ########## Discriminator stuff ##########
        optimizer_D.zero_grad()
        # see how well the discriminator can discriminate
        real_loss = adversarial_loss(discriminator(real_input), valid)
        fake_loss = adversarial_loss(discriminator(fake_output.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        # gradient descent
        d_loss.backward()
        optimizer_D.step()
        
        # progress prints and checkpointing (checkpointing not implemented)
        if epoch % (num_epochs // 10) == 0:
            print("[Epoch %d/%d] [Discriminator Loss: %f] [Generator Loss: %f]" % 
                  (epoch, num_epochs, d_loss.item(), g_loss.item()))
        
        if epoch % (num_epochs // 10) == 0:
            num_samples_to_test = 200
            z = Variable(Tensor(np.random.normal(0, 1, (num_samples_to_test, latent_dim))))
            np_samples = generator(z).cpu().detach().numpy()
            x_sampled = [sample[0] for sample in np_samples]
            y_sampled = [sample[1] for sample in np_samples]
            plt.title("2D Grid of samples obtained")
            plt.scatter(x, y)
            plt.scatter(x_sampled, y_sampled)
            plt.show()
    if save_model:
        torch.save(generator.state_dict(), filename)
    return generator

def load_model(filename, latent_dim=2, img_shape=(2,)):
    generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
    generator.load_state_dict(torch.load(filename))
    if cuda:
        generator.cuda()
    return generator

def classification_function(point):
    x, y = point[0], point[1]
    def send_to_nearest_even(x):
        p1 = int(x)
        if p1 % 2 == 0:
            return p1
        if p1 >= 1:
            return p1 + 1
        return p1 - 1
    return send_to_nearest_even(x), send_to_nearest_even(y)

def point_to_index(point):
    closest = classification_function(point)
    i = int(closest[0]) / 2 + grid_length // 2
    j = int(closest[1]) / 2 + grid_length // 2
    return int(5 * i + j)
