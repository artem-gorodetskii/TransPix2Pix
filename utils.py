import numpy as np
import time
import sys


def num_params(model):
    """
    Counts the number of model parameters.
    
    :model: model, generator or discriminator
    :return: int, the number of model parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

                                                                                            
def adjust_learning_rate(current_iter, optimizer, init_lr, gamma, list_of_iters):
    current_lr = 0
    power = 0
    if current_iter < list_of_iters[0]:
        current_lr = init_lr
    elif current_iter > list_of_iters[-1]:
        current_lr = init_lr * (gamma ** len(list_of_iters))
    else:
        list_of_iters.sort(reverse=True)
        nearest_smaller_iter = min(list_of_iters, key=lambda x : x - current_iter > 0 )
        list_of_iters.sort(reverse=False)
        index_of_nearest_smaller_iter = list_of_iters.index(nearest_smaller_iter) 
        power = index_of_nearest_smaller_iter + 1
        current_lr = init_lr * (gamma ** power)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return current_lr

   
def stream(message):
    """
    Prints the given information in the commad line.
    """
    try:
        sys.stdout.write("\r{%s}" % message)
    except:
        #Remove non-ASCII characters from message
        message = ''.join(i for i in message if ord(i)<128)
        sys.stdout.write("\r{%s}" % message)


class ValueWindow():
    """
    Averages the given values.
    """
    def __init__(self, window_size=100):
        # number of values to average
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []
