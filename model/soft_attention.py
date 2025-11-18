import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """ this function is used to achieve the channel attention module"""
    def __init__(self, in_dim=1023, out_dim=1024, ratio=4):
        super(ChannelAttention, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=out_dim // ratio, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels= out_dim // ratio, out_channels=out_dim, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        out1 = torch.mean(x, dim=-1, keepdim=True)
        out1 = self.mlp(out1)

        out2 = nn.AdaptiveMaxPool1d(1)(x)
        out2 = self.mlp(out2)
        
        out = F.normalize(out1+out2, p=2, dim=1)
        attention = self.sigmoid(out)
        
        return attention