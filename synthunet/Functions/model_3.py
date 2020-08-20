import torch
from torch.nn import functional as F
import torch.nn as nn
class downsample(nn.Module):
    def __init__(self, input_size, output_size, k_size, normalization = True):
        super(downsample, self).__init__()
        self.a = normalization
        self.c1 = nn.Conv3d(in_channels=input_size, out_channels=output_size, kernel_size=k_size,
                            stride=2, padding=2)

        self.norm = nn.BatchNorm3d(num_features=output_size)
        self.relu = nn.LeakyReLU(0.02, True)
    def forward(self,x) :
        x = self.c1(x)
        if self.a:
            x = self.norm(x)
        x = self.relu(x)
        return x


class Adisc(nn.Module):
    def __init__(self, input_size, output_size):
        self.d1 = downsample(input_size=input_size)