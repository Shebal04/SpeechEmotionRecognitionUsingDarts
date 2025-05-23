# search/model_search.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from search.operations import OPS

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS:
            op = OPS[primitive](C, stride)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    def __init__(self, C):
        super(Cell, self).__init__()
        self.op1 = MixedOp(C, stride=1)
        self.op2 = MixedOp(C, stride=1)

    def forward(self, x, weights):
        x1 = self.op1(x, weights[0])
        x2 = self.op2(x, weights[1])
        return x1 + x2

class Network(nn.Module):
    def __init__(self, C=32, num_classes=8, layers=6):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers

        # Initial stem (simple convolution to project features)
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        # Cells
        self.cells = nn.ModuleList()
        for i in range(layers):
            cell = Cell(C)
            self.cells.append(cell)

        # Global pooling and final classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, num_classes)

        # Initialize architecture parameters (alphas)
        self._initialize_alphas()

    def _initialize_alphas(self):
        k = 2  # Two edges per cell (op1 and op2)
        num_ops = len(OPS)
        self.alphas = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [self.alphas]

    def arch_parameters(self):
        return self._arch_parameters

    def forward(self, x):
        s = self.stem(x)

        weights = F.softmax(self.alphas, dim=-1)

        for cell in self.cells:
            s = cell(s, weights)

        out = self.global_pooling(s)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits
