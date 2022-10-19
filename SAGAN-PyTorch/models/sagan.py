# %%
'''
sagan structure.
where is similar with the SAGAN net structure.
code similar sample from the pytorch code, and with the spectral normalization.
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from models.attention import Attention

import numpy as np
# %%
class Generator(nn.Module):
    '''
    pure Generator structure
    '''    
    def __init__(self, image_size=64, z_dim=100, conv_dim=64, channels = 1):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.z_dim = z_dim

        repeat_num = int(np.log2(self.image_size)) - 3  # 3
        mult = 2 ** repeat_num  # 8

        self.l1 = nn.Sequential(
            # input is Z, going into a convolution.
            spectral_norm(nn.ConvTranspose2d(self.z_dim, conv_dim * mult, 4, 1, 0, bias=False),),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(True)
        )

        curr_dim = conv_dim * mult

        self.l2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )
        self.l3 = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(int(curr_dim / 2), int(curr_dim / 4), 4, 2, 1, bias=False),
            ),
            nn.BatchNorm2d(int(curr_dim / 4)),
            nn.ReLU(True),
        )
        self.l4 = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(int(curr_dim / 4), int(curr_dim / 8), 4, 2, 1, bias=False),
            ),
            nn.BatchNorm2d(int(curr_dim / 8)),
            nn.ReLU(True)
        )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(int(curr_dim / 8), self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # self.attn1 = Attention(128)
        self.attn2 = Attention(64)
        # self.attn = Attention(32)

    def forward(self, z):
        out = self.l1(z) # (*, 512, 4, 4)
        out = self.l2(out) # (*, 256, 8, 8)
        out = self.l3(out) # (*, 128, 16, 16)
        # out = self.attn1(out)
        out = self.l4(out) # (*, 64, 32, 32)
        out = self.attn2(out)

        out = self.last(out) # (*, c, 64, 64)

        return out

# %%
class Discriminator(nn.Module):
    '''
    pure discriminator structure

    '''
    def __init__(self, image_size = 64, conv_dim = 64, channels = 1):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        # (*, 1, 64, 64)
        self.l1 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(self.channels, conv_dim, 4, 2, 1, bias=False),
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # (*, 64, 32, 32)
        self.l2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False),
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # (*, 128, 16, 16)
        self.l3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(conv_dim*2, conv_dim * 4, 4, 2, 1, bias=False),
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # (*, 256, 8, 8)
        self.l4 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(conv_dim*4, conv_dim * 8, 4, 2, 1, bias=False),
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # self.attn1 = Attention(256)
        self.attn2 = Attention(512)

        # output layers
        # (*, 512, 4, 4)
        self.last_adv = nn.Sequential(
            spectral_norm(nn.Conv2d(conv_dim*8, 1, 4, 1, 0, bias=False),)
            # without sigmoid, used in the loss funciton
            )

    def forward(self, x):
        out = self.l1(x) # (*, 64, 32, 32)
        out = self.l2(out) # (*, 128, 16, 16)
        out = self.l3(out) # (*, 256, 8, 8)
        # out = self.attn1(out)
        out = self.l4(out) # (*, 512, 4, 4)
        out =  self.attn2(out)

        validity = self.last_adv(out) # (*, 1, 1, 1)

        return validity.squeeze()