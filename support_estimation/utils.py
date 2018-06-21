import numpy as np
import torch
import matplotlib.pyplot as plt
import sys 
import os
sys.path.append(os.path.abspath("/home/nsm/sandbox/PyTorch-GAN"))
import implementations.wgan_gp.wgan_gp as wgp
# from implementations.wgan_gp import Generator

def load_gan(filename):
    model = wgp.Generator()
    model.load_state_dict(torch.load(filename))
    return model

print(wgp.__dict__)
load_gan(None)

