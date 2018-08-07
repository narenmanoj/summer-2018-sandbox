
from  importlib import reload
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

import wae
import schelotto_wae_gan as swg

cuda = torch.cuda.is_available()

print("Cuda: " + str(cuda))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

enc1 = wae.Encoder()
enc2 = swg.Encoder()
dec1 = wae.Decoder()
dec2 = swg.Decoder()
print("nsm implementation")
print(dec1)
print("schelotto implementation")
print(dec2)
