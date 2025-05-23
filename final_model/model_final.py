# final_model/model_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from search.genotypes import load_genotype_from_file, example_genotype

# Create an operations dictionary for the different operations in your architecture
OPS = {
    'identity': lambda C, stride: Identity(),
    'avg_pool': lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1),
    'max_pool': lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    'conv_3x3': lambda C, stride: Conv(C, C, 3, stride),
    'conv_5x5': lambda C, stride: Conv(C, C, 5, stride),
    'lstm': lambda C, stride: RNNCell(C, C, 'lstm'),
    'gru': lambda C, stride: RNNCell(C, C, 'gru'),
    'none': lambda C, stride: Zero(stride)
}

# Basic building blocks
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.
        # Downsample by stride using slicing
        return x[:, :, ::self.stride, ::self.stride] * 0.

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding=None):
        super(Conv, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class RNNCell(nn.Module):
    def __init__(self, C_in, C_out, cell_type='lstm'):
        super(RNNCell, self).__init__()
        # Convert 2D feature maps to sequence for RNN processing
        self.conv_reduce = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(C_out)
        
        # Choose RNN cell type
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(C_out, C_out, batch_first=True)
        elif cell_type == 'gru':
            self.rnn = nn.GRU(C_out, C_out, batch_first=True)
        else:
            raise ValueError("Unsupported RNN cell type")
            
        # Restore feature map dimension
        self.conv_restore = nn.Conv2d(C_out, C_out, 1, bias=False)
        self.bn_restore = nn.BatchNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Reduce channels
        x = self.relu(self.bn_reduce(self.conv_reduce(x)))
        
        # Get dimensions
        batch, channels, height, width = x.size()
        
        # Convert to sequence: [batch, channels, height, width] -> [batch, height*width, channels]
        x = x.view(batch, channels, -1).permute(0, 2, 1)
        
        # Apply RNN
        x, _ = self.rnn(x)
        
        # Convert back to feature map: [batch, height*width, channels] -> [batch, channels, height, width]
        x = x.permute(0, 2, 1).view(batch, channels, height, width)
        
        # Restore feature map
        x = self.relu(self.bn_restore(self.conv_restore(x)))
        return x

# Cell module to handle mixed operations based on genotype
class Cell(nn.Module):
    def __init__(self, genotype, C_prev, C, reduction):
        super(Cell, self).__init__()
        
        # Store if this is a reduction cell
        self.reduction = reduction
        
        # Select the appropriate part of the genotype
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
            
        # Store information
        self.indices = indices
        self.concat = concat
        self.C = C
        
        # Set stride based on whether it's a reduction cell
        stride = 2 if reduction else 1
        
        # Create a preprocessing layer to adjust input channels to the correct size
        self.preprocess = nn.Sequential(
            nn.Conv2d(C_prev, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )
        
        # Create mixed operations based on the genotype
        self.ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C, stride)
            self.ops.append(op)
        
        # Calculate output channels
        self.output_channels = C * len(concat)

    def forward(self, x):
        # Preprocess the input to get consistent channel size
        x = self.preprocess(x)
        original_size = x.size()
        
        # Apply operations based on genotype structure
        states = [x]  # Start with the input state
        
        # Apply operations and accumulate states
        for i, (op, idx) in enumerate(zip(self.ops, self.indices)):
            s = states[idx]  # Get the state to operate on
            s = op(s)  # Apply the operation
            
            # For spatial dimension consistency:
            # If we're in a reduction cell and this isn't the reduced state,
            # we might need to adjust dimensions
            if self.reduction and s.size(2) != states[0].size(2) // 2:
                # Apply average pooling to reduce spatial dimensions
                s = F.interpolate(s, size=(states[0].size(2) // 2, states[0].size(3) // 2), 
                                 mode='bilinear', align_corners=False)
            
            # For spatial dimension consistency with previous states
            elif not self.reduction and s.size(2) != states[0].size(2):
                # Resize to match the spatial dimensions of the first state
                s = F.interpolate(s, size=(states[0].size(2), states[0].size(3)), 
                                 mode='bilinear', align_corners=False)
                
            states.append(s)  # Append to the list of states
        
        # Ensure all states to concatenate have the same spatial dimensions
        states_to_concat = []
        target_size = states[self.concat[0]].size()[2:]
        
        for i in self.concat:
            s = states[i]
            if s.size()[2:] != target_size:
                s = F.interpolate(s, size=target_size, mode='bilinear', align_corners=False)
            states_to_concat.append(s)
        
        # Concatenate the resized states
        return torch.cat(states_to_concat, dim=1)

# Network class for the final architecture
class Network(nn.Module):
    def __init__(self, C=16, num_classes=10, layers=8, genotype=None):
        super(Network, self).__init__()
        self._C = C  # Initial number of channels
        self._num_classes = num_classes
        self._layers = layers
        
        # Use provided genotype or try to load from file (with fallback to example_genotype)
        if genotype is None:
            try:
                genotype = load_genotype_from_file('best_genotype.txt')
                print("Network initialized with genotype from 'best_genotype.txt'")
            except FileNotFoundError:
                genotype = example_genotype
                print("Network initialized with example_genotype (file not found)")
        
        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )
        
        # Build the network using cells
        C_prev = C
        C_curr = C
        self.cells = nn.ModuleList()
        
        for i in range(layers):
            # Every third layer is a reduction cell
            reduction = i in [layers//3, 2*layers//3]
            
            # Double the channels after reduction
            if reduction:
                C_curr *= 2
                
            # Create and add the cell
            cell = Cell(genotype, C_prev, C_curr, reduction)
            self.cells.append(cell)
            
            # Update for next layer
            C_prev = cell.output_channels
        
        # Global average pooling and classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        # Initial stem
        x = self.stem(x)
        
        # Pass through all cells
        for cell in self.cells:
            x = cell(x)
            
        # Global pooling and classification
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        
        return logits