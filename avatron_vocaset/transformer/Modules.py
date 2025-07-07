import torch
import torch.nn as nn
import numpy as np
from frontend.modules import DepthwiseConv
import pdb

class ConvBlock(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(3):
            conv = nn.Sequential(nn.Conv1d(in_dim, in_dim//2, 3, padding=1),
                                 nn.ReLU(), 
                                 nn.BatchNorm1d(in_dim//2)
                                )
            self.enc.append(conv)
            in_dim = in_dim//2

        for i in range(3):
            conv = nn.Sequential(nn.Conv1d(in_dim, in_dim*2, 3, padding=1),
                                 nn.ReLU(), 
                                 nn.BatchNorm1d(in_dim*2)
                                )
            self.dec.append(conv)
            in_dim = in_dim*2


    def forward(self, x):
        x = x.permute(0, 2, 1)
        for conv in self.enc:
            x = conv(x)

        for conv in self.dec:
            x = conv(x)
        x = x.transpose(0,1)
        return x