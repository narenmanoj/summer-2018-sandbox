import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

import numpy as np
import matplotlib.pyplot as plt
import sys 
import os
sys.path.append(os.path.relpath("../pytorch_gan"))
import implementations.wgan_gp.wgan_gp as wgp
# from implementations.wgan_gp import Generator

def load_gan(filename, gan_flavor):
    model = gan_flavor.Generator()
    model.load_state_dict(torch.load(filename))
    return model

def show_images(images):
    npimg = images.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def sample_from_gan(my_gan, num_images):
    assert num_images > 1
    z = Variable(Tensor(np.random.normal(0, 1, (num_images, 100))))
    fake_imgs = my_gan(z)
    return fake_imgs

def view_samples(my_gan, num_images, nrow=0):
    if nrow == 0:
        nrow = int(np.sqrt(num_images))
    fake_imgs = sample_from_gan(my_gan, num_images)
    show_images(make_grid(fake_imgs.data[:num_images], nrow=nrow, normalize=True))
