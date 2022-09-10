import torch
import torch.nn as nn


class WeightEMA(object):
    """
    Computes the exponential moving average of the model weights.
    """
    def __init__(self, model, ema_model, lr, alpha=0.999):
        # the model itself 
        self.model = model
        
        # its copy with smoothed parameters
        self.ema_model = ema_model
        
        # smoothimg parameter
        self.alpha = alpha
   
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)


    def step(self):
        """
        Computes one step of exponential moving average smoothimg.
        """
        one_minus_alpha = 1.0 - self.alpha
        for i, p in enumerate(zip(self.params, self.ema_params)):
            param = p[0]
            ema_param = p[1]
            if 'int' not in str(ema_param.dtype):
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
            else:
                ema_param = param.clone()
                