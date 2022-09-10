import torch


class LossNetwork(torch.nn.Module):
    """
    Reference: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    
    This class performs VGG16 Loss Network computations for feature reconstruction loss.
    """
    def __init__(self, vgg_model):
        """
        :vgg_model: VGG16 model
        :return: relu2_2 output
        """
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '8': "relu2_2",
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output.get('relu2_2')
        