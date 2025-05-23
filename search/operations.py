# search/operations.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)

class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(AvgPool, self).__init__()
        self.op = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False)

    def forward(self, x):
        return self.op(x)

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MaxPool, self).__init__()
        self.op = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.op(x)

class SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True):
        super(SeqLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        # Flatten spatial dimensions
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, sequence_len, features)
        x, _ = self.lstm(x)
        # reshape back to (B, hidden_dim*2, H', W') approximately
        out_dim = int((H*W)**0.5)
        x = x.permute(0, 2, 1)
        x = x.view(B, x.size(1), out_dim, out_dim)
        return x

class SeqGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True):
        super(SeqGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)
        x, _ = self.gru(x)
        out_dim = int((H*W)**0.5)
        x = x.permute(0, 2, 1)
        x = x.view(B, x.size(1), out_dim, out_dim)
        return x

# List of all possible operations
OPS = {
    'none'      : lambda C, stride: Zero(),
    'identity'  : lambda C, stride: Identity(),
    'avg_pool'  : lambda C, stride: AvgPool(3, stride, 1),
    'max_pool'  : lambda C, stride: MaxPool(3, stride, 1),
    'conv_3x3'  : lambda C, stride: ReLUConvBN(C, C, 3, stride, 1),
    'conv_5x5'  : lambda C, stride: ReLUConvBN(C, C, 5, stride, 2),
    'lstm'      : lambda C, stride: SeqLSTM(C, C),
    'gru'       : lambda C, stride: SeqGRU(C, C),
}
