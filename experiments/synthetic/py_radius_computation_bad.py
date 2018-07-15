
# coding: utf-8

# In[1]:


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

import gan

cuda = torch.cuda.is_available()

print("Cuda: " + str(cuda))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# In[2]:


# load the model back and compute our measurement
latent_dim = 2
loaded_gen = gan.load_model("ok_generator_81", layer_width=3).eval()

for param in loaded_gen.parameters():
    param.requires_grad = False

print(gan.class_radius(loaded_gen, verbose=False))


# In[ ]:


results = []
for i in range(10):
    x = gan.class_radius(loaded_gen, verbose=False, num_trials=100)
    results.append(x)
    print(x)

print("Final results: " + str(results))
