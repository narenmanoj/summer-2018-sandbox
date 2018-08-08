from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument(
    '--batch-size',
    type=int,
    default=150,
    metavar='N',
    help='input batch size for training (default: 150)')
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    metavar='N',
    help='number of epochs to train (default: 10)')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='enables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--clip', type=float, default=-1, metavar='N', help='weight clip limit')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs)

def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=1, bn=True, relu=True):
    block = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]
    if bn:
        block.append(nn.BatchNorm2d(out_channels))
    if relu:
        block.append(nn.ReLU(True))
        
def upconv_block(in_channels, out_channels, kernel_size, stride=1, padding=1, bn=True, relu=True):
    block = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]
    if bn:
        block.append(nn.BatchNorm2d(out_channels))
    if relu:
        block.append(nn.ReLU(True))

class DCGanVAE(nn.Module):
    def __init__(self):
        super(DCGanVAE, self).__init__()
        filter_size = 4
        stride = 2
        same = (filter_size - stride) // 2
        
        self.model = nn.Sequential(*conv_block(1, 128, filter_size, stride=stride, padding=same),
                                   *conv_block(128, 256, filter_size, stride=stride, padding=same),
                                   *conv_block(256, 512, filter_size, stride=stride, padding=same))
        
        self.fc1 = nn.Linear(, 20) 
        
    def encode(self, x):
        out = self.model(x)
        return self.fc1(out)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        pass
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.enc_layers = [self.fc1, self.fc21, self.fc22]

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        if args.clip >= 1e-5:
            for layer in model.enc_layers:
                for p in layer.parameters():
                    p.data.clamp_(-args.clip, args.clip)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)

            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n],
                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]
                ])
                save_image(
                    comparison.cpu(),
                    'results/reconstruction_%d.png' % epoch,
                    nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def test_interpolate(epoch, num_intermediates=20, num_interpolations=1):
    ni = num_intermediates
    n = num_interpolations
    if n <= 0:
        return
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            for j in range(0, args.batch_size, 2):
                # interpolate between j and j + 1
                mu1, logvar1 = model.encode(data[j].view(-1, 784))
                mu2, logvar2 = model.encode(data[j + 1].view(-1, 784))
                z1 = model.reparameterize(mu1, logvar1)
                z2 = model.reparameterize(mu2, logvar2)
                test_imgs = [
                    model.decode((1 - (lam / ni * 1.0)) * z1 +
                                 (lam / ni * 1.0) * z2)[0] for lam in range(ni)
                ]
                test_imgs = [img.view(1, 1, 28, 28) for img in test_imgs]
                test_imgs_tensor = torch.cat(test_imgs)
                clamp = args.clip if args.clip > 0 else 0
                save_image(test_imgs_tensor.cpu(),
                           'results/interpolation_%d_%d_%f.png' % (epoch, n, clamp))
            n -= 1
            if n <= 0:
                break


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    test_interpolate(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28), 'results/sample_%d.png' % epoch)

for layer in model.enc_layers:
    for p in layer.parameters():
        print(torch.max(torch.abs(p.data)))