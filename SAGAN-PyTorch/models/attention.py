import torch
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np
from torch.nn.utils import spectral_norm 

def conv1x1(in_channels, out_channels): # not change resolusion
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query_conv = nn.utils.spectral_norm(conv1x1(channels, channels // 8))#spectral_norm光谱归一化 query
        self.key_conv = nn.utils.spectral_norm(conv1x1(channels, channels // 8))#key
        #value：
        self.g = nn.utils.spectral_norm(conv1x1(channels, channels // 2))
        self.o = nn.utils.spectral_norm(conv1x1(channels // 2, channels))
        ##可学习的参数
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, inputs):
        batch, c, h, w = inputs.size()

        query_conv = self.query_conv(inputs) # (*, c/8, h, w)
        key_conv = F.max_pool2d(self.key_conv(inputs), [2, 2]) # (*, c/8, h/2, w/2)
        g = F.max_pool2d(self.g(inputs), [2, 2]) # (*, c/2, h/2, w/2)

        query_conv = query_conv.view(batch, self.channels // 8, -1) # (*, c/8, h*w)
        key_conv = key_conv.view(batch, self.channels // 8, -1) # (*, c/8, h*w/4)
        g = g.view(batch, self.channels // 2, -1) # (*, c/2, h*w/4)

        beta = F.softmax(torch.bmm(query_conv.transpose(1, 2), key_conv), -1) # (*, h*w, h*w/4)
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(batch, self.channels//2, h, w)) # (*, c, h, w)

        return self.gamma * o + inputs

