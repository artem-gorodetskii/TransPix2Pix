import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """
    Basic convolutional block of discriminator.
    """
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )


    def forward(self, x):
        
        return self.conv(x)


class Discriminator(nn.Module):
    """
    Model discriminator. 
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"), 
            nn.LeakyReLU(0.2),
        ).apply(initialize_weights)

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

        self.model = nn.Sequential(*layers).apply(initialize_weights)


    def forward(self, x, y):
        """
        Computes the forward pass of the discriminator.
        
        :x: torchTensor, sketch 
        :y: torchTensor, generated image or ground truth image
        :return: torchTensor, result of discriminator forward pass
        """
        # concatenate sketch with generated image or ground truth image
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)           
        x = self.model(x)            
        return x

    
def initialize_weights(m):
    """
    Initializes model weights with a Gaussian distribution with mean 0 and standard deviation 0.02.
    
    :m: model, discriminator or genertor
    """
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
