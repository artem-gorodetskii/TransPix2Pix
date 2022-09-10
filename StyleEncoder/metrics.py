import torch
import torch.nn as nn


class Accuracy(nn.Module):
    """
    This class is for computing accuracy.
    """
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        correct = 0
        total = 0
        _, predicted = torch.max(y_pred.data, 1)
        total += y_true.size(0)
        correct += (predicted == y_true).sum().item()
        acc = 100 * correct / total

        return acc
        