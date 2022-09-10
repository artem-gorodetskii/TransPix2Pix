import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils import tensorboard
from torch.utils.data import DataLoader
import random
import yaml
import glob2
import argparse
import time
import copy
import pandas as pd
from pathlib import Path 
from ast import literal_eval
from evaluator import Evaluator
from model_ema import WeightEMA
from loss_network import LossNetwork
from model_generator import Generator
from data_generator import CatsDataset
from model_discriminator import Discriminator, initialize_weights
from utils import ValueWindow, stream, num_params, adjust_learning_rate


class Trainer:
    """
    The class contains the necessary functions for the neural network training.
    """
    def __init__(self, args, args_yaml):
        """
        :args: paresed arguments
        :args_yaml: arguments loaded from 'config.yaml'
        """
        # Initialize some training details
        self.LOAD_MODEL = args.LOAD_MODEL
        self.iter_load = args.iter_load
        self.device = args_yaml['device'] if torch.cuda.is_available() else 'cpu'
        self.BCE = nn.BCEWithLogitsLoss()
        self.L1_LOSS = nn.L1Loss()
        self.MSE_LOSS = nn.MSELoss()
        self.learning_rate = args_yaml['learning_rate']
        self.weight_decay = args_yaml['weight_decay']
        self.grad_clip = args_yaml['grad_clip']
        self.L1_lambda = args_yaml['L1_lambda']
        self.num_epochs = args_yaml['num_epochs']
        self.save_model = args_yaml['save_model']    
        self.eval_every =  args_yaml['eval_every']
        self.save_every = args_yaml['save_every']
        self.norm_type = args_yaml['norm_type']
        self.milestones_disc = args_yaml['milestones_disc']
        self.milestones_gen = args_yaml['milestones_gen']
        self.gamma = args_yaml['gamma']
        
        # Initialize and load VGG16 model used for feature loss
        self.vgg_model = models.vgg.vgg16(pretrained=True)
        if torch.cuda.is_available():
            self.vgg_model.cuda()
        self.loss_network = LossNetwork(self.vgg_model)
        self.loss_network.eval()
        
        # Initialize Discriminator and Generator models
        self.disc = Discriminator(in_channels=3).to(self.device)
        self.gen = Generator(in_channels=3, norm_type=self.norm_type).to(self.device)
        
        # Initialize optimizers wiht weight decay only for certain parameters
        params_disc = []
        for name, values in self.disc.named_parameters():
            if 'bias' not in name and 'conv.1.weight' not in name:
                params_disc += [{'params': [values], 'lr': self.learning_rate, 'weight_decay': self.weight_decay}]
            else:
                params_disc += [{'params': [values], 'lr': self.learning_rate, 'weight_decay': 0.0}]
        
        params_gen = []
        for name, values in self.gen.named_parameters():
            if 'bias' not in name and 'cSE' not in name and 'sSE' not in name and 'normalize' not in name and 'bottleneck' not in name and 'encoder_layer' not in name and 'transformer_encoder' not in name and 'pos_embedding' not in name and 'encoder_layer' not in name and 'transformer_encoder' not in name and 'ConvBlock1.1.weight' not in name and 'ConvBlock2.2.weight' not in name and 'ConvBlock2.1.weight' not in name and 'Encoder.10.'  not in name and 'fcl3' not in name and '.1.weight' not in name and 'BatchNorm' not in name:
                print(name)
                params_gen += [{'params': [values], 'lr': self.learning_rate, 'weight_decay': self.weight_decay}]
            else:
                params_gen += [{'params': [values], 'lr': self.learning_rate, 'weight_decay': 0.0}]
     
        self.opt_disc = optim.Adam(params_disc, lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_gen = optim.Adam(params_gen, lr=self.learning_rate, betas=(0.5, 0.999))  
        
        # Initialize Discriminator and Generator models for exponential moving average
        self.EMA_disc = Discriminator(in_channels=3).to(self.device)
        self.EMA_gen = Generator(in_channels=3, norm_type=self.norm_type).to(self.device)
        
        for param in self.EMA_disc.parameters():
            param.detach_()
        for param in self.EMA_gen.parameters():
            param.detach_()
        
        # Initialize exponential moving average 'optimizer' to perform moving average of EMA_disc and EMA_gen parameters
        self.EMA_opt_disc = WeightEMA(self.disc, self.EMA_disc, lr=self.learning_rate, alpha=0.98)
        self.EMA_opt_gen = WeightEMA(self.gen, self.EMA_gen, lr=self.learning_rate, alpha=0.98) 

        # Initialize evaluation class
        self.evaluator = Evaluator(args_yaml, self.EMA_disc, self.EMA_gen, self.eval_every)
        
        # Count the number of models parametrs
        params_disc, params_gen = num_params(self.disc), num_params(self.gen)
        print(f'gen params = {params_gen}, disc params = {params_disc}')
            

    def save_checkpoint(self, iter, checkpoints_dir):
        """
        Saves current generator and discriminator checkpoints.
        
        :last_train_step: int, number of the current iteration
        :checkpoints_dir: str, path to save generator and discriminator checkpoints
        """
        print("=> Saving checkpoint ...")
        checkpoint_disc = {
            "state_dict": self.disc.state_dict(),
            "optimizer": self.opt_disc.state_dict(),
        }
        checkpoint_gen = {
            "state_dict": self.gen.state_dict(),
            "optimizer": self.opt_gen.state_dict(),
        }
        torch.save(checkpoint_disc, checkpoints_dir.joinpath(f'disc_iter_{iter}.pth'))
        torch.save(checkpoint_gen, checkpoints_dir.joinpath(f'gen_iter_{iter}.pth'))
        
        checkpoint_EMA_disc = {
            "state_dict": self.EMA_disc.state_dict()
        }
        checkpoint_EMA_gen = {
            "state_dict": self.EMA_gen.state_dict()
        }
        torch.save(checkpoint_EMA_disc, checkpoints_dir.joinpath(f'EMA_disc_iter_{iter}.pth'))
        torch.save(checkpoint_EMA_gen, checkpoints_dir.joinpath(f'EMA_gen_iter_{iter}.pth'))
        print("=> Checkpoint was saved!")
 

    def load_checkpoint(self, iter_load, checkpoints_dir):
        """
        Loads generator and discriminator checkpoints.
        
        :iter_load: int, number of the checkpoint iteration to load 
        :checkpoints_dir: str, path to the checkpoints
        """
        print(f"Loading checkpoint of {self.iter_load} iter ...")
        checkpoint_disc_dir = checkpoints_dir.joinpath(f'disc_iter_{self.iter_load}.pth')
        checkpoint_gen_dir = checkpoints_dir.joinpath(f'gen_iter_{self.iter_load}.pth')
        checkpoint_disc = torch.load(checkpoint_disc_dir, map_location='cpu')
        checkpoint_gen = torch.load(checkpoint_gen_dir, map_location='cpu')
        self.disc.load_state_dict(checkpoint_disc["state_dict"])
        self.gen.load_state_dict(checkpoint_gen["state_dict"])
        self.opt_disc.load_state_dict(checkpoint_disc["optimizer"])
        self.opt_gen.load_state_dict(checkpoint_gen["optimizer"])
        
        checkpoint_EMA_disc_dir = checkpoints_dir.joinpath(f'EMA_disc_iter_{self.iter_load}.pth')
        checkpoint_EMA_gen_dir = checkpoints_dir.joinpath(f'EMA_gen_iter_{self.iter_load}.pth')
        checkpoint_EMA_disc = torch.load(checkpoint_EMA_disc_dir, map_location='cpu')
        checkpoint_EMA_gen = torch.load(checkpoint_EMA_gen_dir, map_location='cpu')
        self.EMA_disc.load_state_dict(checkpoint_EMA_disc["state_dict"])
        self.EMA_gen.load_state_dict(checkpoint_EMA_gen["state_dict"])
        print('Checkpoint was restored!')
        

    def feature_loss(self, y_output, y_content):
        """
        Compute feature reconstruction loss.  
        
        :y_output: torchTensor, generated image
        :y_content: torchTensor, ground truth image
        """
        self.loss_network.eval()
        with torch.no_grad():
            out = self.MSE_LOSS(self.loss_network(y_output), self.loss_network(y_content)) 
            
        return out


    def one_step(self, x, y, embedding):
        """
        Performs one training step (one iteration) for discriminator and generator.
        
        :x: torchTensor, sketch
        :y: torchTensor, ground truth image
        :return: (float, float), loss functions value of the discriminator and generator
        """
        self.disc.train()
        self.gen.train()
        
        x = x.to(self.device, dtype=torch.float)  
        y = y.to(self.device, dtype=torch.float)
        embedding = embedding.to(self.device)  
        
        self.opt_disc.zero_grad()
        self.opt_gen.zero_grad()

        # Train discriminator
        y_fake = self.gen(x, embedding)
        D_real = self.disc(x, y)
        D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
        D_fake = self.disc(x, y_fake.detach())
        D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        D_loss.backward()
        self.opt_disc.step()
        self.EMA_opt_disc.step()

        # Train generator
        D_fake = self.disc(x, y_fake)
        G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
        L1 = self.L1_LOSS(y_fake, y) * self.L1_lambda
        G_feauter_loss = self.feature_loss(y_output=y_fake, y_content=y)
        G_loss = G_fake_loss + L1 + G_feauter_loss
        
        G_loss.backward()
        
        # Clip gradient norm of an iterable of parameters
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.gen.parameters(), self.grad_clip, norm_type=2.0)
            
        self.opt_gen.step()
        self.EMA_opt_gen.step()
        
        return D_loss.item(), G_loss.item()


    def train(self, run_id: str, models_dir: str, max_step: int, train_loader, val_loader):
        """
        Performs network training during the given numder of epochs.
        
        :run_id: int, identification number of training
        :models_dir: str, path to save all training materials
        :max_step: int, number of maximum iteration
        :train_loader: DataLoader object with training examples
        :val_loader: DataLoader object with validation examples
        """
        # Create the required directories 
        models_dir = Path(models_dir)
        models_dir.mkdir(exist_ok=True)
        model_dir = models_dir.joinpath(run_id)
        model_dir.mkdir(exist_ok=True)
        examples_dir = model_dir.joinpath('examples') 
        tb_directory =  model_dir.joinpath('tb_events') 
        checkpoints_dir = model_dir.joinpath('checkpoints')          
        examples_dir.mkdir(exist_ok=True)
        tb_directory.mkdir(exist_ok=True) 
        checkpoints_dir.mkdir(exist_ok=True) 
        
        step = 0
        start_time = time.time()
        
        # Load checkpoint
        if self.LOAD_MODEL:
            self.load_checkpoint(self.iter_load, checkpoints_dir)
            
        current_epoch = self.gen.get_step() // len(train_loader)
        
        # Create ValueWindow for discriminator and generator loss
        D_loss_window = ValueWindow(100)
        G_loss_window = ValueWindow(100)
        time_window = ValueWindow(100)
        
        # Create tensorboard
        tensorboard_writer = tensorboard.SummaryWriter(log_dir=tb_directory)
        
        # Training loop with epochs
        for epoch in range(current_epoch, self.num_epochs + 1):
            # Training loop with iterations
            for x, y, embedding in train_loader:
                start_time_iter = time.time()
                step = self.gen.get_step()
                
                # Get currant learning rate 
                current_lr_disc = adjust_learning_rate(step, self.opt_disc, self.learning_rate, self.gamma, self.milestones_disc)
                current_lr_gen = adjust_learning_rate(step, self.opt_gen, self.learning_rate, self.gamma, self.milestones_gen)
                
                # Perform one training step
                d_loss, g_loss = self.one_step(x, y, embedding)     
                D_loss_window.append(d_loss)
                G_loss_window.append(g_loss)         
                
                step = self.gen.get_step()
                last_train_step = copy.deepcopy(step)
                
                # Models evaluation
                if self.eval_every != 0 and last_train_step % self.eval_every == 0:
                    val_d_loss, val_g_loss = self.evaluator.evaluate(val_loader, step, examples_dir, self.loss_network) #степ поменялся
                    
                    self.gen.set_step(last_train_step)
                    
                    # Write training information on tensorboard
                    tensorboard_writer.add_scalar(tag='D_loss/validation', scalar_value=val_d_loss, global_step=last_train_step)
                    tensorboard_writer.add_scalar(tag='G_loss/validation', scalar_value=val_g_loss, global_step=last_train_step)
                    
                    tensorboard_writer.add_scalar(tag='D_loss/train', scalar_value=D_loss_window.average, global_step=last_train_step)
                    tensorboard_writer.add_scalar(tag='G_loss/train', scalar_value=G_loss_window.average, global_step=last_train_step)
                    
                    tensorboard_writer.add_scalar(tag='lr_disc/train', scalar_value=current_lr_disc, global_step=last_train_step)
                    tensorboard_writer.add_scalar(tag='lr_gen/train', scalar_value=current_lr_gen, global_step=last_train_step)
                 
                
                # Stop training if maximum iteration is reached
                if step >= max_step:
                    break
                
                # Print training details
                time_window.append(time.time() - start_time_iter)
                current_time = time.time() - start_time
                msg = f'Epoch: {epoch}/{self.num_epochs} | Step: ({step}/{max_step}) | D: {D_loss_window.average:#.3}, G: {G_loss_window.average:#.4} | {1./time_window.average:#.2} steps/s | Time: {current_time / 60 :#.3}'
                
                stream(msg)
            
                # Save models checkpoints
                if last_train_step != 0 and last_train_step % self.save_every == 0:
                    self.save_checkpoint(last_train_step, checkpoints_dir)
                

if __name__ == "__main__":

    # Parse and read arguments
    parser = argparse.ArgumentParser(description='Enter the required parameters and paths')

    parser.add_argument('--run_id', type=str, help='Identification number of training')
    parser.add_argument('--LOAD_MODEL', default=False, type=bool, help='Load the pretrained model if True')
    parser.add_argument('--iter_load', type=str, help='Number of the checkpoint to load')

    args = parser.parse_args()

    with open(r'config.yaml') as file:
        args_yaml = yaml.load(file)
    
    # Read csv file with embeddings obtained from Style-Pre-Net
    df = pd.read_csv('dataset/data_color_emb256.csv',converters={"label_onehot": literal_eval, 'embedding': literal_eval})
    
    # Load data paths, spit into validation and training sets
    dataset_path = args_yaml['DATASET_PATH']
    data_paths = glob2.glob(f'{dataset_path}/*.png')

    random.seed(0)

    path_val = random.sample(data_paths, int(len(data_paths) * 0.06))
    path_train = [file for file in data_paths if file not in path_val] 
    
    # Initialilize training/validtion datasets and dataloaders
    train_dataset = CatsDataset(path_train, dataframe=df, augment=False)
    train_loader = DataLoader(train_dataset, 
                              batch_size=args_yaml['batch_size'], 
                              shuffle=True, num_workers=args_yaml['num_workers'],  
                              drop_last=True)

    val_dataset = CatsDataset(path_val,  dataframe=df, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    # Compute number of maximum training step
    max_step = args_yaml['num_epochs'] * len(train_loader)
    
    # Path for saving all training materials including model checkpoint, generated examples, tensorboard data
    models_dir = args_yaml['models_dir']
    
    # Initialize and run training loop
    trainer = Trainer(args, args_yaml)
    trainer.train(args.run_id, models_dir = models_dir, max_step=max_step, train_loader=train_loader, val_loader=val_loader)
    