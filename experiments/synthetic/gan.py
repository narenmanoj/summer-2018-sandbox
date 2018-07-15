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

# defining generator and discriminator architecture, taken from Github Pytorch-GAN (need to modify hyperparams)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(2,), layer_width=400, grid_length=5):
        super(Generator, self).__init__()
        
        self.img_shape = img_shape
        self.grid_length = grid_length
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.0, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, layer_width, normalize=True),
            *block(layer_width, layer_width),
            *block(layer_width, layer_width),
            *block(layer_width, layer_width),
            nn.Linear(layer_width, int(np.prod(img_shape))),
        )
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return torch.clamp(img, min=-self.grid_length, max=self.grid_length)
    
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

def train(save_model=False, filename="", num_samples=10000, 
          num_epochs=10000, num_samples_per_batch=500, 
          grid_length=5, var=0.0025, latent_dim=2, layer_width=400):
    # num_samples is for plotting purposes
    if save_model:
        assert len(filename) > 0
    
    # initialize things
    generator = Generator(latent_dim=latent_dim, layer_width=layer_width, grid_length=grid_length)
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
            num_samples_to_test = 2000
            z = Variable(Tensor(np.random.normal(0, 1, (num_samples_to_test, latent_dim))))
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

def load_model(filename, latent_dim=2, img_shape=(2,), grid_length=5, layer_width=400):
    generator = Generator(latent_dim=latent_dim, img_shape=img_shape, grid_length=grid_length, layer_width=layer_width)
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

def point_to_index(point, grid_length=5):
    def convert_one(point):
        closest = classification_function(point)
        i = int(closest[0]) // 2 + grid_length // 2
        j = int(closest[1]) // 2 + grid_length // 2
        return int(grid_length * i + j)
    return np.array([np.array([convert_one(p)]) for p in point])

class DifferentiableClassifier(nn.Module):
    def __init__(self, grid_length=5):
        super(DifferentiableClassifier, self).__init__()
        
        self.grid_length = grid_length
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.0, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(2, 100),
            *block(100, 100),
            *block(100, 100),
            *block(100, 100),
            *block(100, 100),
            nn.Linear(100, 1)
        )
        
    def forward(self, p):
        return self.model(p)

def sample_points_plane(low, high, num_points):
    return [(np.random.uniform(low=low, high=high), 
             np.random.uniform(low=low, high=high)) for i in range(num_points)]
    
def train_diffable_classifier(grid_length=5, num_epochs=10000, num_samples_per_batch=5):
    low = -2 * (grid_length // 2) - 2
    high = 2 * (grid_length // 2) + 2
    
    my_classifier = DifferentiableClassifier(grid_length=grid_length)
    if cuda:
        my_classifier.cuda()
    loss = torch.nn.MSELoss()
    learning_rate = 0.0002
    b1 = 0.5
    b2 = 0.999
    amsgrad = True
    optimizer = torch.optim.Adam(my_classifier.parameters(), 
                                 lr=learning_rate, betas=(b1, b2), 
                                 amsgrad=amsgrad)
    for epoch in range(num_epochs):
        # draw some random training point
        p_np = sample_points_plane(low, high, num_samples_per_batch)
        p = Variable(Tensor(p_np))
        answer = Variable(Tensor(point_to_index(p_np, grid_length=grid_length)), requires_grad=False)
        optimizer.zero_grad()
        test_output = my_classifier(p)
        c_loss = loss(test_output, answer)
        c_loss.backward()
        optimizer.step()
        
        if epoch % (num_epochs // 10) == 0:
            print("[Epoch %d] [Loss %f]" % (epoch, c_loss.item()))
            
    torch.save(my_classifier.state_dict(), "differentiable_classifier_5")
    return my_classifier
    
def class_radius_loss(gen, z, r, eps=2, lamb=100):
    first = torch.norm(r, p=2).pow(2)
    constraint = torch.max(torch.abs(gen(r + z) - gen(z))) - eps
    opt_success = False
    if float(constraint) >= 0:
        lamb = 0
        opt_success = True # this is a feasible point
    second = torch.mul(constraint, lamb)
    return first - second, opt_success
    
def dist_to_boundary(gen, z, latent_dim=2, eps=2, lamb=10000, num_iter=20, lr=0.05, verbose=False):
    z.require_grad = False
    initial_std = 0.000025
    r = Variable(Tensor(np.random.normal(0, initial_std, (1, latent_dim))), requires_grad=True)
    optimizer = torch.optim.Adam([r], lr=lr, betas=(0.5, 0.999), amsgrad=True)
    lowest_loss = float(lamb * 2)
    opt_success = False
    for i in range(num_iter):
        if verbose:
            print("[Iteration %d]" % i)
        optimizer.zero_grad()
        loss_val, opt_success_step = class_radius_loss(gen, z, r, eps=eps, lamb=lamb)
        if opt_success_step:
            opt_success = True
        loss_val.backward()
        if lowest_loss > float(loss_val):
            if verbose:
                print("Flipping %f to %f" % (lowest_loss, float(loss_val)))
            lowest_loss = float(loss_val)
            best_r = r.clone()
            if verbose:
                print("Setting r to " + str(best_r))
        optimizer.step()
    
        if verbose:
            print("Regularized loss: %f" % (float(loss_val)))
    if not opt_success:
        return Variable(Tensor(np.zeros((1, latent_dim))), requires_grad=True), opt_success
    return best_r, opt_success
        

def class_radius(gen, latent_dim=2, num_trials=1000, num_sub_trials=10, verbose=False, num_retrials=20):
    ans = 0
    dividend = num_trials
    for i in range(num_trials):
        if i % (num_trials // 20) == 0:
            print("Iteration %d" % i)
        z = Variable(Tensor(np.random.normal(0, 1, (1, latent_dim))), requires_grad=False)
        dist = np.inf
        for j in range(num_sub_trials):
            include_computation = False
            for k in range(num_retrials):
                r, opt_success = dist_to_boundary(gen, z, latent_dim=latent_dim, lr=0.125, verbose=verbose)
                if opt_success:
                    include_computation = True
                    break
                if k != num_retrials - 1 and verbose:
                    print("RETRYING")
                elif verbose:
                    print("FAILED")
            if not include_computation:
                continue
            d = float(torch.norm(r, p=2))
            dist = min(dist, d)
        if dist == np.inf:
            dividend -= 1
            print("Couldn't compute for the selected z")
            continue
        ans += dist
    return ans / dividend
