import torch
import torch.nn as nn
import numpy as np
from attention_block import SelfAttentionBlock
from transformer_block import TransformerBlock


class Identity(nn.Module):
    """
    Performs the identity transformation.
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
          
class cSE_Block(nn.Module):
    """
    Performs Spatial Squeeze & Channel Excitation.
    """
    def __init__(self, in_channels, r):
        """
        :in_channels: int, number of input channels
        :r: int, the dimensionality reduction ratio 
        """
        super(cSE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        bs, in_channels, _, _ = x.shape
        y = self.squeeze(x).view(bs, in_channels)
        y = self.excitation(y).view(bs, in_channels, 1, 1)
        return x * y.expand_as(x)   
    
    
class sSE_Block(nn.Module):
    """
    Performs Channel Squeeze & Spatial Excitation.
    """
    def __init__(self, in_channels):
        """
        :in_channels: int, number of input channels
        """
        super(sSE_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1, stride=1)


    def forward(self, x):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = input_x * x

        return x

    
class scSE_Block(nn.Module):
    """
    Performs spatial-channel squeeze & excitation.
    """
    def __init__(self, in_channels, r):
        """
        :in_channels: int, number of input channels
        :r: int, the dimensionality reduction ratio 
        """
        super(scSE_Block, self).__init__()

        self.cSE = cSE_Block(in_channels, r)
        self.sSE = sSE_Block(in_channels)


    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)
        x = cSE + sSE

        return x


class DownConvBlock(nn.Module):
    """
    Basic convolution block of generator encoder.
    """
    def __init__(self, in_channels, out_channels, norm: bool, norm_type: str,
                 conv3: bool=False, 
                 c_SE: bool=True, s_SE: bool=True, sc_SE: bool=False, 
                 r_SE: int = 16, bottleneck: bool=False):
        """
        :in_channels: int, number of input channels
        :out_channels: int, number of output channels
        :norm: bool, whether to normalize data or not
        :norm_type: str, type of data normalization (can be 'BatchNorm' or 'GroupNorm')
        :conv3: bool, whether to add third convolution or not
        :c_SE: bool, if True, performs channel squeeze & excitation
        :s_SE: bool, if True, performs spacial squeeze & excitation
        :sc_SE: bool, if True, performs spacial-channel squeeze & excitation
        :r_SE: int, the dimensionality reduction ratio in channel squeeze & excitation
        :bottleneck: bool, if True, initializes the second conv of block as Conv2d with 1x1 kernel size
        """
        super().__init__()
        self.Identity = Identity()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels) if norm == True and norm_type == 'BatchNorm' else Identity(),
            nn.GroupNorm(32, out_channels) if norm == True and norm_type == 'GroupNorm' else Identity(),
            nn.LeakyReLU(0.2))
            
        if not bottleneck:
            self.ConvBlock2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels) if norm == True and norm_type == 'BatchNorm' else Identity(),
                nn.GroupNorm(32, out_channels) if norm == True and norm_type == 'GroupNorm' else Identity(),
                nn.LeakyReLU(0.2))
        else:
            self.ConvBlock2_bottleneck = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels) if norm == True and norm_type == 'BatchNorm' else Identity(),
                nn.GroupNorm(32, out_channels) if norm == True and norm_type == 'GroupNorm' else Identity(),
                nn.LeakyReLU(0.2))

        self.conv3 = conv3

        if self.conv3:
            self.ConvBlock3 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels) if norm == True and norm_type == 'BatchNorm' else Identity(),
                nn.GroupNorm(32, out_channels) if norm == True and norm_type == 'GroupNorm' else Identity(),
                nn.LeakyReLU(0.2))

            self.ConvBlock3_bottleneck = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels) if norm == True and norm_type == 'BatchNorm' else Identity(),
                nn.GroupNorm(32, out_channels) if norm == True and norm_type == 'GroupNorm' else Identity(),
                nn.LeakyReLU(0.2))
        
        # Inittialize normalization layer
        self.norm = norm
        if self.norm:
            if norm_type == 'BatchNorm':
                self.normalize = nn.BatchNorm2d(out_channels)
            else:
                self.normalize = nn.GroupNorm(32, out_channels)
        
        # Inittialize activation layer
        self.act = nn.LeakyReLU(0.2)
        
        # Inittialize squeeze & excitation
        self.c_SE = c_SE
        self.s_SE = s_SE
        self.sc_SE = sc_SE

        if self.c_SE:
            self.cSE = cSE_Block(out_channels, r_SE)
        if self.s_SE:
            self.sSE = sSE_Block(out_channels)
        if self.sc_SE:
            self.scSE = scSE_Block(out_channels, r_SE)
        

    def forward(self, x):
        
        out = self.ConvBlock1(x)
        identity_1 = out 
        
        if out.shape[2] == 2: 
            out = nn.functional.pad(out, (1,1,1,1), mode ='reflect')
            out = self.ConvBlock2(out)
        elif out.shape[2] == 1: 
            out = self.ConvBlock2_bottleneck(out)
        else:    
            out = nn.functional.pad(out, (1,2,2,1), mode ='reflect')
            out = self.ConvBlock2(out)
            
        # Spatial Squeeze & Channel Excitation
        if self.c_SE:             
            out = self.cSE(out)
        
        # residual connection
        out = out + identity_1
        
        # Channel Squeeze & Spatial Excitation
        if self.s_SE:             
            out = self.sSE(out)
        
        # Concurrent squeeze & excication
        if self.sc_SE:           
            out = self.scSE(out)
            
        if self.norm:
            out = self.normalize(out)
        out = self.act(out)
        
        # Additional third convolution
        if self.conv3:        
            identity_2 = out 
            
            if out.shape[2] == 2: 
                out = nn.functional.pad(out, (1,1,1,1), mode ='reflect')
                out = self.ConvBlock3(out)
            elif out.shape[2] == 1: 
                out = self.ConvBlock3_bottleneck(out)
            else:
                out = nn.functional.pad(out, (1,2,2,1), mode ='reflect')
                out = self.ConvBlock3(out)
            
            out = out + identity_1 + identity_2
            
            if self.norm:
                out = self.normalize(out)
            out = self.act(out)
       
        return out

    
class UpConvBlock(nn.Module):
    """
    Basic convolution block of generator decoder.
    """
    def __init__(self, in_channels, out_channels, norm_type: str,
                 dropout: bool, 
                 conv3: bool=False, 
                 c_SE: bool=True, s_SE: bool=True, sc_SE: bool=False, 
                 r_SE: int = 16, bottleneck: bool=False, convbottleneck: bool=False, k: float=1.33):
        """
        :in_channels: int, number of input channels
        :out_channels: int, number of output channels
        :norm_type: str, type of data normalization (can be 'BatchNorm' or 'GroupNorm')
        :dropout: bool, whether to use DropOut or not
        :conv3: bool, whether to add third convolution or not
        :c_SE: bool, if True, performs channel squeeze & excitation
        :s_SE: bool, if True, performs spacial squeeze & excitation
        :sc_SE: bool, if True, performs spacial-channel squeeze & excitation
        :r_SE: int, the dimensionality reduction ratio in channel squeeze & excitation
        :bottleneck: bool, if True, initializes the second conv of block as Conv2d with 1x1 kernel size
        :convbottleneck: bool, if True, initializes the ConvBottleneck block
        :k: float, reduction factor of ConvBottleneck
        """
        super().__init__()
        self.Identity = Identity()
        self.ConvBlock1 = nn.Sequential(
                                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels) if norm_type == 'BatchNorm' else nn.GroupNorm(32, out_channels),
                                        nn.ReLU()
                                        )
        
        self.convbottleneck = convbottleneck
        # Initialize ConvBottleneck
        if self.convbottleneck:
            self.ConvBottleneck1 = nn.Sequential(
                                                 nn.Conv2d(out_channels, int(out_channels / k), kernel_size=1, stride=1, 
                                                           padding=0, bias=False, padding_mode="reflect"),
                                                 nn.BatchNorm2d(int(out_channels / k)) if norm_type == 'BatchNorm' else Identity(),
                                                 nn.GroupNorm(32, out_channels) if norm_type == 'GroupNorm' else Identity(),
                                                 nn.ReLU()
                                                 )

            if bottleneck:
                self.ConvBottleneck2 = nn.Sequential(
                                                     nn.Conv2d(int(out_channels / k),  int(out_channels / k), kernel_size=1, 
                                                                stride=1, padding=0, bias=False, padding_mode="reflect"),
                                                     nn.BatchNorm2d(int(out_channels / k)) if norm_type == 'BatchNorm' else Identity(),
                                                     nn.GroupNorm(32, out_channels) if norm_type == 'GroupNorm' else Identity(),
                                                     nn.ReLU()
                                                     )
            else:
                self.ConvBottleneck2 = nn.Sequential(
                                                     nn.Conv2d(int(out_channels / k), int(out_channels / k), kernel_size=4,
                                                     stride=1, padding=0, bias=False, padding_mode="reflect"),
                                                     nn.BatchNorm2d(int(out_channels / k)) if norm_type == 'BatchNorm' else Identity(),
                                                     nn.GroupNorm(32, out_channels) if norm_type == 'GroupNorm' else Identity(),
                                                     nn.ReLU()
                                                     )

            self.ConvBottleneck3 = nn.Sequential(nn.Conv2d(int(out_channels / k), out_channels, kernel_size=1, stride=1, 
                                                           padding=0, bias=False, padding_mode="reflect"),
                                                 nn.BatchNorm2d(out_channels) if norm_type == 'BatchNorm' else Identity(),
                                                 nn.GroupNorm(32, out_channels) if norm_type == 'GroupNorm' else Identity(),
                                                 nn.ReLU()
                                                 ) 
        
        if bottleneck:
            self.ConvBlock2 = nn.Sequential(
                                            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, 
                                                      padding=0, bias=False, padding_mode="reflect"),
                                            nn.BatchNorm2d(out_channels) if norm_type == 'BatchNorm' else Identity(),
                                            nn.GroupNorm(32, out_channels) if norm_type == 'GroupNorm' else Identity(),
                                            nn.ReLU()
                                            )
        else:
            self.ConvBlock2 = nn.Sequential(
                                            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, 
                                                      padding=0, bias=False, padding_mode="reflect"),
                                            nn.BatchNorm2d(out_channels) if norm_type == 'BatchNorm' else Identity(),
                                            nn.GroupNorm(32, out_channels) if norm_type == 'GroupNorm' else Identity(),
                                            nn.ReLU()
                                            )
          
        self.conv3 = conv3

        if self.conv3:
            self.ConvBlock3 = nn.Sequential(
                                            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, 
                                                      padding=0, bias=False, padding_mode="reflect"),
                                            nn.BatchNorm2d(out_channels) if norm_type == 'BatchNorm' else nn.GroupNorm(32, out_channels),
                                            nn.ReLU())
        
        # Initialize normalization layer 
        if norm_type == 'BatchNorm':
            self.normalize = nn.BatchNorm2d(out_channels)
        else:
            self.normalize = nn.GroupNorm(32, out_channels)
            
        # Initialize activation layer     
        self.act = nn.ReLU() 
        
        # Initialize dropout layer 
        self.dropout = dropout
        if self.dropout:
            self.Dropout = nn.Dropout(0.3)
        
        # Initialize squeeze & excication
        self.c_SE = c_SE
        self.s_SE = s_SE
        self.sc_SE = sc_SE

        if self.c_SE:
            self.cSE = cSE_Block(out_channels, r_SE)
        if self.s_SE:
            self.sSE = sSE_Block(out_channels)
        if self.sc_SE:
            self.scSE = scSE_Block(out_channels, r_SE)
        

    def forward(self, x):
        out = self.ConvBlock1(x)

        # ConvBottleneck
        if self.convbottleneck:
            # 1) Conv1x1: C x H x W -> C/k x H x W
            out = self.ConvBottleneck1(out)
            if out.shape[2] == 2: 
                # 2) Conv1x1: C/k x H x W -> C/k x H x W
                out = self.ConvBottleneck2(out)
            else:
                # 2) Conv4x4: C/k x H x W -> C/k x H x W
                out = nn.functional.pad(out, (1,2,2,1), mode ='reflect')
                out = self.ConvBottleneck2(out)
                
            # 3) Conv1x1: C/k x H x W -> C x H x W    
            out = self.ConvBottleneck3(out)
        
        identity_1 = out

        if out.shape[2] == 2: 
            out = self.ConvBlock2(out)
        else:    
            out = nn.functional.pad(out, (1,2,2,1), mode ='reflect')
            out = self.ConvBlock2(out)
        
        if self.c_SE:             
            out = self.cSE(out)
        
        out = out + identity_1
        
        if self.s_SE:             
            out = self.sSE(out)
        
        if self.sc_SE:            
            out = self.scSE(out) 
        
        out = self.normalize(out)    
        out = self.act(out)
        
        # Additional third convolution
        if self.conv3:        
            identity_2 = out 
            
            if out.shape[2] == 2:
                out = nn.functional.pad(out, (1,1,1,1), mode ='reflect')
                out = self.ConvBlock3(out)
            else:
                out = nn.functional.pad(out, (1,2,2,1), mode ='reflect')
                out = self.ConvBlock3(out)
            
            out = out + identity_1 + identity_2
            
            out = self.normalize(out)
            out = self.act(out)
        
        if self.dropout:
            out = self.Dropout(out)
            
        return out

    
class StylePreNet(nn.Module):
    """
    This class is the Style-Pre-Net module.
    """
    def __init__(self, in_features,dropout=0.4):
        super().__init__()
        """
        :in_features: int, number of input features of style embedding
        :return: splitted into 8 parts style embedding 
        """
        self.fcl1 = nn.Linear(in_features, in_features*4)
        self.fcl2 = nn.Linear(in_features*4, in_features*4)
        self.fcl3 = nn.Linear(in_features*4, in_features)
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)        
        self.emb_size = 256
        self.num_parts = 8
        self.part_len = self.emb_size // self.num_parts
        

    def forward(self, embedding):

        bs = embedding.shape[0]
        x = self.fcl1(embedding)
        x = self.act(x)
        x = self.do(x)
        x = self.fcl2(x)
        x = self.act(x)
        x = self.do(x)
        x = self.fcl3(x)
        x = x.view(bs, self.num_parts, self.part_len, 1, 1)

        return x


class Final(nn.Module):
    """
    Final convolution block of generator decoder, that maps from 32 channels to 3 channels using a Conv 1x1 and Tanh activation function.
    """
    def __init__(self, in_channels, out_channels):
        super(Final, self).__init__()
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.final_act = nn.Tanh()
    
    def forward(self, x):
        """
        :x: torchTensor, output of generator decoder
        :return: torchTensor, generated photorealistic cat image
        """
        x = self.final_conv(x)
        x = self.final_act(x)
        
        return x
           
                              
class Generator(nn.Module):    
    """
    Model generator.
    """
    def __init__(self, 
                 in_channels=3, 
                 features: list=[64, 128, 256, 512, 512, 512, 512, 512], 
                 transf_enc_pos: list=[4, 5, 6], 
                 norm_type: str='BatchNorm',
                 emb_len: int=32,
                ):
        """
        :in_channels: int, number of input channels
        :features: list of ints, numbers of output feature maps of DownConvBlocks
        :transf_pos: list of ints, the number of DownConvBlock, after which to add the TransformerBlock 
        :norm_type: str, type of data normalization (can be 'BatchNorm' or 'GroupNorm')
        :emb_len: int, part length of style embedding
        """
        super().__init__()   
        self.in_channels = in_channels
        self.Dropout = nn.Dropout(0.3)
        self.Dropout_skip = nn.Dropout(0.3)
        
        # List of encoder blocks
        blocks_down = []
        
        # Initialize encoder blocks
        for i, feature in enumerate(features):

            # Do not normalize data in the first encoder convolution block
            if i == 0:
                norm = False
            else:
                norm = True 
            
            if i == len(features) - 1:
                bottleneck = True
            else:
                bottleneck = False 
            
            if i+1 in transf_enc_pos:
                # Add DownConvBlock to list and initialize its weights
                blocks_down.append(DownConvBlock(in_channels, feature, norm=norm, norm_type=norm_type, 
                                                 bottleneck=bottleneck).apply(initialize_weights))   

                # Add TransformerBlock to list in respective position after DownConvBlock
                blocks_down.append(TransformerBlock(d_model=feature, position=i+1))
                in_channels = feature

            else:
                # Add DownConvBlock to list and initialize its weights
                blocks_down.append(DownConvBlock(in_channels, feature, norm=norm, norm_type=norm_type, 
                                                 bottleneck=bottleneck).apply(initialize_weights))
                in_channels = feature
        
        # Calculate the TransformerBlock positional indices in the encoder
        self.transf_pos_in_enc = [i*2-4 for i in transf_enc_pos]

        # Calculate the Self-AttentionBlock positional indices in the decoder
        self.transf_dec_pos = [len(features) - pos -1 for pos in list(reversed(transf_enc_pos))]
        self.transf_pos_in_dec = [i*2 for i in self.transf_dec_pos]    
        
        # List of decoder blocks
        blocks_up = [] 
        
        # Initialize decoder blocks
        for i, feature in enumerate(reversed(features[:-1])):
            
            # Use dropout only for first three UpConvBlocks
            if i < 3: 
                dropout = True
            else:
                dropout = False
                
            # Fisrt UpConvBlock does not concatenate with skip connection
            if i == 0:
                # add UpConvBlock to list and initialize its weights
                blocks_up.append(UpConvBlock(in_channels + emb_len, 
                                             feature, 
                                             dropout=dropout, 
                                             norm_type=norm_type, 
                                             bottleneck=True).apply(initialize_weights))
                in_channels = feature
            else:
                if i in self.transf_dec_pos:
                    # Add UpConvBlock to list and initialize its weights
                    blocks_up.append(UpConvBlock(in_channels*2 + emb_len, 
                                                 feature, 
                                                 dropout=dropout, 
                                                 norm_type=norm_type).apply(initialize_weights)
                    )
                    # Add Self-AttentionBlock to list in respective position after UpConvBlock
                    blocks_up.append(SelfAttentionBlock(position=len(features) - 1 - i, 
                                                        hid_dim=feature))
                    in_channels = feature
                else:
                    if i == self.transf_dec_pos[-1] + 1:
                        # Add UpConvBlock with ConvBottleneck module to list and initialize its weights
                        blocks_up.append(UpConvBlock(in_channels*2 + emb_len, 
                                                     feature, 
                                                     dropout=dropout, 
                                                     norm_type=norm_type, 
                                                     convbottleneck=True, k=1.6).apply(initialize_weights))
                        in_channels = feature 

                    elif i == self.transf_dec_pos[-1] + 2:
                        # Add UpConvBlock with ConvBottleneck module to list and initialize its weights
                        blocks_up.append(UpConvBlock(in_channels*2 + emb_len, 
                                                     feature, 
                                                     dropout=dropout, 
                                                     norm_type=norm_type, 
                                                     convbottleneck=True, k=1.33).apply(initialize_weights)
                        )
                        in_channels = feature 
                    else:
                        # Add UpConvBlock module to list and initialize its weights
                        blocks_up.append(UpConvBlock(in_channels*2 + emb_len, 
                                                     feature, 
                                                     dropout=dropout, 
                                                     norm_type=norm_type).apply(initialize_weights))
                        in_channels = feature              
        
        # Add last UpConvBlock to list and initialize its weights
        blocks_up.append(UpConvBlock(features[0]*2 + emb_len, 32, 
                                     dropout=False, norm_type=norm_type).apply(initialize_weights))
        
        # Create encoder
        self.Encoder = nn.Sequential(*blocks_down)
        
        # Create decoder
        self.Decoder = nn.Sequential(*blocks_up) 
        
        # Initialize StylePreNet
        self.StylePreNet = StylePreNet(256)
        
        # Initialize final convolution block of decoder and its and initial weights
        self.final = Final(32, 3).apply(initialize_weights)
        
        # Initialize 'step' variable - number of generator forward pass
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        

    def forward(self, x, embedding):
        # Count the number of forward pass
        self.step += 1

        out = x
        # Create a python list for skip connections 
        results_down = []
        
        for i, BlockDown in enumerate(self.Encoder):  
          
            if i not in self.transf_pos_in_enc:
                out = BlockDown(out)
                # Add skip connection to list
                results_down.append(self.Dropout_skip(out))
                out = self.Dropout(out)
            else:
                out = BlockDown(out)
        
        # Compute style embedding
        color_embedding = self.StylePreNet(embedding)
        
        # Reverse skip connection list for consistent concatenation
        results_down = list(reversed(results_down))  
        concat_idx = 1
        for i, BlockUp in enumerate(self.Decoder):
            if i == 0:
                h, w = results_down[i].shape[2], results_down[i].shape[3]
                # Repeat style embedding parts values along the spatial dimension
                out = BlockUp(torch.cat([color_embedding[:, i, :, :, :].repeat(1,1,h,w), results_down[i]], 1))
            else:
                if i in self.transf_pos_in_dec:
                    out, att = BlockUp(out) 
                else:
                    h, w = results_down[concat_idx].shape[2], results_down[concat_idx].shape[3]
                    # Repeat style embedding parts values along the spatial dimension, concatenate with skip connection
                    out = BlockUp(torch.cat([out,
                                             color_embedding[:, concat_idx, :, :, :].repeat(1,1,h,w), 
                                             results_down[concat_idx]], 1)) 
                    concat_idx+=1

        return self.final(out)
                         
    def get_step(self):
        return self.step.data.item()
      
    def set_step(self, value):
        self.step = self.step.data.new_tensor([value])
        
        
def initialize_weights(m):
    """
    Initializes model weights with a Gaussian distribution with mean 0 and standard deviation 0.02.
    
    :m: model
    """
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
