import torch
import torch.nn as nn
import torch.optim as optim


class TransformerBlock(nn.Module):
    """
    This class performs TransformerBlock.
    """
    def __init__(self,
                 d_model: int, 
                 position: int,
                 nhead=8, 
                 dropout=0.1, 
                 num_layers=6,
                 ): 
        super(TransformerBlock, self).__init__()
        """
        :d_model: int, the number of expected features in the input 
        :position: int, position of TransformerBlock in decoder, used to compute the spatial size of feature map  
        :nhead: int, the number of heads in the Multi-head Attention
        :num_layers: int, the number of Transformer encoder layers in the TransformerBlock
        """
        self.img_size = 256 // (2 ** position) # the spatial size of feature map 
        self.seq_len =  self.img_size  ** 2  # seq_len = hight * widht
        self.hid_dim = d_model
        self.dim_feedforward = self.hid_dim * 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, self.dim_feedforward, dropout) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.pos_embedding = nn.Embedding(self.seq_len, d_model)
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(self.device)
        self.dropout = nn.Dropout(0.1)
        self.BatchNorm = nn.BatchNorm2d(d_model)
        
    def reshape(self, x): # input shape [ batch size, features, height, width]
        """
        Performs processing to obtain the sequence.
        
        :x: torchTensor, output of DownConvBlock
        
        :return: torchTensor, sequence of shape [height * width, batch size, features]
        """
        x = x.flatten(2) # [batch size, features, height * width]
        x = x.transpose(-1, -2) # [batch size, height * width, features]
        x = x.transpose(-2, -3)  # [height * width, batch size, features]
        
        return x

    def forward(self, x):
        x = self.reshape(x)
        batch_size = x.shape[1]

        pos = torch.arange(0, self.seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        z = self.dropout((x * self.scale) + self.pos_embedding(pos).transpose(-2, -3))

        out = self.transformer_encoder(z)
        out =  torch.reshape(out, (self.img_size, self.img_size, batch_size, self.hid_dim)).permute(2, 3, 0, 1)
        out = self.BatchNorm(out)
            
        return out
