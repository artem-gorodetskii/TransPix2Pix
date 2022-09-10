import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import pandas as pd
from ast import literal_eval
from skimage import io, transform
from torchvision import models
from loss_network import LossNetwork


class Evaluator:
    """
    This class performs model evaluation.
    """
    def __init__(self, args, disc, gen, eval_every):
        self.device = args['device'] if torch.cuda.is_available() else 'cpu'
        self.eval_every = eval_every
        
        # Initialize discriminator and generator models
        self.disc = disc
        self.gen = gen
        
        # Loss functions
        self.BCE = nn.BCEWithLogitsLoss()
        self.L1_LOSS = nn.L1Loss()
        self.MSE_LOSS = nn.MSELoss()
        self.L1_LAMBDA = args['L1_lambda']
        self.dataframe = pd.read_csv('data_color_emb256.csv',converters={"label_onehot": literal_eval, 'embedding': literal_eval})


    def save_some_examples(self, val_loader, step, examples_dir: str):
        """
        Saves the evaluation example of generated cat.
        
        :val_loader: DataLoader object
        :step: int, number of current iteration
        :examples_dir: str, path to save the evaluation example of generated cat
        """
        x, y, embedding = next(iter(val_loader))
        x, y, embedding = x.to(self.device), y.to(self.device), embedding.to(self.device)
        self.gen.eval()
        with torch.no_grad():
            # Generator inference
            y_fake = self.gen(x, embedding)
            y_fake = y_fake * 0.5 + 0.5 
            
            # Save generated image
            save_image(y_fake, examples_dir.joinpath(f'y_gen_{step}.png')) 
            
            # Save sketch and ground truth images only on the first evaluation step
            if step == self.eval_every:
                save_image(self.x * 0.5 + 0.5, examples_dir.joinpath(f'input_{step}.png')) 
                save_image(self.y * 0.5 + 0.5, examples_dir.joinpath(f'label_{step}.png')) 

    
    def feature_loss(self, y_output, y_content, network):
        """
        Compute feature reconstruction loss.  
        
        :y_output: torchTensor, generated image
        :y_content: torchTensor, ground truth image
        :network: loss network
        """
        network.eval()
        with torch.no_grad():
            out = self.MSE_LOSS(network(y_output), network(y_content)) 
            
        return out
    

    def one_step(self, x, y, embedding, loss_network):
        """ 
        Computes the discriminator and generator loss functions value on one validation data instance.
        
        :x: torchTensor, sketch
        :y: torchTensor, ground truth image
        :return: (float, float), loss functions value of the discriminator and generator
        """
        self.disc.eval()
        self.gen.eval()
        with torch.no_grad():
            x = x.to(self.device, dtype=torch.float)  
            y = y.to(self.device, dtype=torch.float)  
            embedding = embedding.to(self.device)
            
            y_fake = self.gen(x, embedding)
            
            D_real = self.disc(x, y)
            D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
            D_fake = self.disc(x, y_fake)
            D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            
            D_fake = self.disc(x, y_fake)
            G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
            L1 = self.L1_LOSS(y_fake, y) * self.L1_LAMBDA
            G_feauter_loss = self.feature_loss(y_output=y_fake, y_content=y, network=loss_network)
            G_loss = G_fake_loss + L1 + G_feauter_loss

        return D_loss.item(), G_loss.item()


    def evaluate(self, val_loader, step, examples_dir: str, loss_network):
        """ 
        Computes the average value of the discriminator and generator loss functions on the whole validation set.

        :val_loader: DataLoader object
        :step: int, number of current iteration
        :return: (float, float), average value of the discriminator and generator loss functions
        """
        d_losses = []
        g_losses = []
        for x, y, embedding in val_loader:
            d_loss, g_loss = self.one_step(x, y, embedding, loss_network)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
        final_d_loss =  np.array(d_losses).mean()
        final_g_loss =  np.array(g_losses).mean()
        print(f' val_D = {final_d_loss:.3f}, val_G = {final_g_loss:.3f}') 
        
        # Save trainig examples 
        self.save_some_examples(val_loader, step, examples_dir)
        
        return final_d_loss, final_g_loss
    