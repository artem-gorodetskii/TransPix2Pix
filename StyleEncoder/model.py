import torch
import torch.nn as nn


class InitalConvBlock(nn.Module):
    """
    This class performs the initial convolutional block of Style Encoder.
    """
    def __init__(self, in_channels,  out_channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True).apply(initialize_weights_he)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.spatial_do = nn.Dropout2d(dropout / 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True).apply(initialize_weights_he)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        residual = x
        
        x = self.spatial_do(x)
        x = self.conv2(x)
        x = self.norm2(x)
        
        x = x + residual
        
        x = self.act2(x)
        x = self.do(x)
        
        return x
    

class ConvBlock(nn.Module):
    """
    This class performs 2,3,4 convolutional blocks of Style Encoder.
    """
    def __init__(self, in_channels, out_channels, dropout, last_dropout):
        super().__init__()
        """
        :last_dropout: bool, if True, use different last dropout probability 
        """
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True).apply(initialize_weights_he)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True).apply(initialize_weights_he)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.spatial_do = nn.Dropout2d(dropout / 2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True).apply(initialize_weights_he)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU()
        
        if last_dropout:
            self.do = nn.Dropout(0.5)
        else:
            self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        residual_1 = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        x = x + residual_1
        
        x = self.act2(x)
        
        residual_2 = x
        
        x = self.spatial_do(x)
        x = self.conv3(x)
        x = self.norm3(x)
        
        x = x + residual_2
        
        x = self.act3(x)
        x = self.do(x)

        return x         

    
def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)
 
    
class StyleEncoder(nn.Module):
    """
    This class is the Style Encoder.
    """
    def __init__(self, init_channels=32, num_features=256, num_classes=9, dropout=0.4, s=5, m=0.30):
        super().__init__()
        """
        :init_channels: int, initial value of feature filters number
        :num_features: int, 
        :num_classes: int, size of style embedding
        :s: float, CosFace layer scaling parameter, fixed value of 'x' norm 
        :m: float, CosFace layer parameter, margin
        
        :return: torchTensor, style embedding
        """
        self.s = s
        self.m = m
        self.init_block = InitalConvBlock(3, init_channels, dropout)
        self.block1 = ConvBlock(init_channels, init_channels*2, dropout, last_dropout=False)
        self.block2 = ConvBlock(init_channels*2, init_channels*4, dropout, last_dropout=False)
        self.block3 = ConvBlock(init_channels*4, init_channels*8, dropout, last_dropout=True)
        
        self.norm1 = nn.BatchNorm2d(init_channels*8)
        self.fcl1 = nn.Linear(init_channels*8 * 8*8, num_features* 4).apply(initialize_weights_he)
        self.fcl_do = nn.Dropout(0.1)
        self.fcl2 = nn.Linear(num_features* 4, num_features).apply(initialize_weights_he)
        self.norm2 = nn.BatchNorm1d(num_features)
        
        self.weight = nn.Parameter(torch.Tensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        
    def forward(self, x, label_idx):
        self.step += 1
        
        x = self.init_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.norm1(x)
        x = torch.flatten(x, 1)
        x = self.fcl1(x)
        x = self.fcl_do(x)
        out_fcl = self.fcl2(x)              
        x = self.norm2(out_fcl)
        
        cosine = cosine_sim(x, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label_idx.view(-1, 1), 1.0)
        out = self.s * (cosine - one_hot * self.m)
        
        return out, out_fcl

    def get_step(self):
        return self.step.data.item()
      
    def set_step(self, value):
        self.step = self.step.data.new_tensor([value])
    
    
def initialize_weights_he(m):
    for name, param in m.named_parameters():
        if 'bias' in name:
            torch.nn.init.zeros_(param.data)
        else:
            torch.nn.init.kaiming_normal_(param.data)
    
        
def initialize_weights_x(m):
    for name, param in m.named_parameters():
        if 'bias' in name:
            torch.nn.init.zeros_(param.data)
        else:
            torch.nn.init.xavier_uniform_(param.data)
            