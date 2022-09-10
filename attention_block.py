import torch
import torch.nn as nn
import torch.optim as optim


class SelfAttentionBlock(nn.Module):
    """
    This class performs Dot-Product Self-Attention.
    """
    def __init__(self, position, hid_dim=512, n_heads=1):
        super().__init__()
        """
        :position: int, position of MultiHeadAttentionLayer in decoder, used to compute the spatial size of feature map  
        :hid_dim: int, the number of expected features in the input
        :n_heads: int, the number of heads in the multiheadattention models
        """
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads # the number of expected features for every head
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(self.device) #scale factor to reduce position embeddings influence
        self.img_size = 256 // (2 ** position) # the spatial size of feature map  
        self.seq_len =  self.img_size  ** 2  # sequence length
        self.pos_embedding = nn.Embedding(self.seq_len, hid_dim)
        self.BatchNorm = nn.BatchNorm2d(self.hid_dim)
        

    def preprocessing(self, x, add_pos: bool=True): # input shape: [batch size, features, height, width]
        """
        Performs processing to obtain the sequence.
        
        :x: torchTensor, output of UpConvBlock
        :add_pos: bool, if True, add positional embedding
        
        :return: torchTensor, sequence of shape [batch size, height * width, hid_dim]
        """
        x = x.flatten(2) # [batch size, features, height * width]
        x = x.transpose(-1, -2) # [batch size, height * width, features]
        x = x.transpose(-2, -3)  # [height * width, batch size, features]
        
        batch_size = x.shape[1]
        if add_pos:
            pos = torch.arange(0, self.seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) # [batch size, height * width]
            x = x * self.scale_factor + self.pos_embedding(pos).transpose(-2, -3).to(self.device) # [height * width, batch size, hid_dim]
            
        x = x.transpose(1, 0)  # [batch size, height * width, hid_dim]
        
        return x
        
        
    def forward(self, x): # [batch size, hid dim, height, width]
        
        batch_size = x.shape[0]
        x = self.preprocessing(x, add_pos=False) # [batch size, query len, hid dim]

        Q = x.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  #[batch size, n heads, query len, head dim]
        K = x.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = x.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
                       
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) 
        
        #energy = [batch size, n heads, query len, key len]
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
       
        x = torch.matmul(attention, V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = x.view(batch_size, self.hid_dim, self.img_size,  self.img_size)
        
        #x = [batch size, img_size, img_size, hid dim]
        
        x = self.BatchNorm(x)
        
        return x, attention
        