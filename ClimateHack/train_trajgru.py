import torch
import torch.nn as nn
import torch.optim as optim
from numpy import float32
#from torch.utils.data import DataLoader, random_split
#from dataset import ClimateHackDataset
from loss import MS_SSIMLoss
import random
# from submission.model import Model
import numpy as np
# import cv2
# from submission.model import *
from submission.forecaster import Forecaster
from submission.encoder import Encoder as Encoder2
from submission.net_params import return_params_trajgru
#from models2.config import cfg
from pytorch_msssim import MS_SSIM
from einops import rearrange
#import pickle
#import pytorch_lightning as pl
#from training_utils import *
#from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class MS_SSIMLoss(nn.Module):
    """Multi-Scale SSIM Loss"""

    def __init__(self, channels=1,**kwargs):
        """
        Initialize
        Args:
            convert_range: Convert input from -1,1 to 0,1 range
            **kwargs: Kwargs to pass through to MS_SSIM
        """
        super(MS_SSIMLoss, self).__init__()
        # self.ssim_module = MS_SSIM(
        #     data_range=255, size_average=True, win_size=3, channel=channels, **kwargs
        # )
        self.ssim_module = MS_SSIM(size_average=True, win_size=3, channel=channels, **kwargs
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward method
        Args:
            x: tensor one
            y: tensor two
        Returns: multi-scale SSIM Loss
        """
        return 1.0 - self.ssim_module(x, y)

class TempModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        encoder_params1, forecaster_params1 = return_params_trajgru(config['innersize'])
        self.encoder = Encoder2(encoder_params1[0], encoder_params1[1])
        self.forecaster = Forecaster(forecaster_params1[0], forecaster_params1[1])
        # self.ef = EF(self.encoder, self.forecaster)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, features):
        x = rearrange(features, 'b (c t) h w -> t b c h w', c=1)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.forecaster(x)
        x = rearrange(x, 't b c h w -> b (t c) h w')
        return x

if __name__ == '__main__':
    #args['innersize'] = list(map(int, args['innersize'].split()))
    #def train():
    #    train_model(args, TempModel, 'trajgru-1')
    #if args['sweep']:
    #    wandb.agent(args['sweepid'], function=train, count=args['sweepruns'], entity="loluwot", project="ClimateHack")
    #else:
    #    train()
    pass