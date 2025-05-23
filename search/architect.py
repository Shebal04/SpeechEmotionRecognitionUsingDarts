# search/architect.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from search.model_search import Network

class Architect:
    def __init__(self, model, lr_arch=3e-4, lr_model=1e-3, weight_decay=3e-4):
        self.model = model
        self.lr_arch = lr_arch
        self.lr_model = lr_model
        self.weight_decay = weight_decay

        # Optimizer for model weights (parameters of the neural network)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr_model, weight_decay=self.weight_decay)

        # Optimizer for architecture parameters (alphas for operations)
        self.arch_optimizer = optim.Adam(self.model.arch_parameters(), lr=self.lr_arch)

    def step(self, input_train, target_train, input_valid, target_valid, weight_optimizer=True):
        """
        Perform one step of bilevel optimization:
        1. Update model weights with respect to architecture parameters.
        2. Update architecture parameters with respect to model weights.
        """

        # Step 1: Update model weights based on architecture parameters
        self.model_optimizer.zero_grad()
        loss_train = self._compute_loss(input_train, target_train)
        loss_train.backward()
        self.model_optimizer.step()

        # Step 2: Update architecture parameters using the validation set
        self.arch_optimizer.zero_grad()
        loss_valid = self._compute_loss(input_valid, target_valid)
        loss_valid.backward()
        self.arch_optimizer.step()

        return loss_train.item(), loss_valid.item()

    def _compute_loss(self, input_data, target_data):
        """
        Compute the cross-entropy loss for a batch of input and target data.
        This will be used to calculate the gradients and update both the model weights and architecture parameters.
        """
        logits = self.model(input_data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, target_data)
        return loss

    def reset_arch_optimizer(self):
        """
        Reset the architecture optimizer (i.e., to perform updates with a fresh state).
        """
        self.arch_optimizer = optim.Adam(self.model.arch_parameters(), lr=self.lr_arch)
        
    def get_model_optimizer(self):
        """
        Return the model's optimizer.
        """
        return self.model_optimizer

    def get_arch_optimizer(self):
        """
        Return the architecture's optimizer.
        """
        return self.arch_optimizer
