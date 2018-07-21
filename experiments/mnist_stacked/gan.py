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


# defining generator and discriminator architecture, taken from Github Pytorch-GAN (need to modify hyperparams)
class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_size=32, channels=3):
        super(Generator, self).__init__()

        self.latent_dim = 100
        self.img_size = 32
        self.channels = 3

        self.init_size = self.img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh())

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCDiscriminator(nn.Module):
    def __init__(self, img_size=32, channels=3):
        super(Discriminator, self).__init__()

        self.img_size = 32
        self.channels = 3

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.img_size // 2**4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def show(img):
    npimg = img.clone().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def get_mnist_dataloader(batch_size, img_size=32):
    assert batch_size % 3 == 0  # for stacking
    os.makedirs("./data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])),
        batch_size=batch_size,
        shuffle=True)
    return dataloader


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
          num_samples=10000,
          num_epochs=10000,
          num_samples_per_batch=500,
          grid_length=5,
          var=0.0025,
          latent_dim=2,
          layer_width=400):
    # num_samples is for plotting purposes
    if save_model:
        assert len(filename) > 0

    # initialize things
    generator = Generator(
        latent_dim=latent_dim,
        layer_width=layer_width,
        grid_length=grid_length)
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
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=8 * learning_rate,
        betas=(b1, b2),
        amsgrad=amsgrad)
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(b1, b2),
        amsgrad=amsgrad)

    # underlay of true distribution
    real_samples = sample_from_2dgrid(
        grid_length=grid_length, num_samples=num_samples)
    x = [real_samples[i][0] for i in range(num_samples)]
    y = [real_samples[i][1] for i in range(num_samples)]

    # train the GAN
    for epoch in range(num_epochs):
        # ground truths
        valid = Variable(
            Tensor(num_samples_per_batch, 1).fill_(1.0), requires_grad=False)
        fake = Variable(
            Tensor(num_samples_per_batch, 1).fill_(0.0), requires_grad=False)

        # input
        real_input = Variable(
            Tensor(
                sample_from_2dgrid(
                    grid_length=grid_length,
                    var=var,
                    num_samples=num_samples_per_batch)))

        ########## Generator stuff ##########
        optimizer_G.zero_grad()
        # sample latent z
        z = Variable(
            Tensor(
                np.random.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(num_samples_per_batch, latent_dim))))

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
            print("[Epoch %d/%d] [Discriminator Loss: %f] [Generator Loss: %f]"
                  % (epoch, num_epochs, d_loss.item(), g_loss.item()))

        if epoch % (num_epochs // 10) == 0:
            num_samples_to_test = 2000
            z = Variable(
                Tensor(
                    np.random.normal(0, 1, (num_samples_to_test, latent_dim))))
            np_samples = generator(z).cpu().detach().numpy()
            x_sampled = [sample[0] for sample in np_samples]
            y_sampled = [sample[1] for sample in np_samples]
            plt.title("2D Grid of samples obtained")
            plt.scatter(x, y)
            plt.scatter(x_sampled, y_sampled)
            plt.show()
    if save_model:
        assert len(filename) > 0
        torch.save(generator.state_dict(), filename)
    return generator


def load_model(filename,
               latent_dim=2,
               img_shape=(2, ),
               grid_length=5,
               layer_width=400):
    generator = Generator(
        latent_dim=latent_dim,
        img_shape=img_shape,
        grid_length=grid_length,
        layer_width=layer_width)
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