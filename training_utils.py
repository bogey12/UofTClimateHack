
from multiprocessing.dummy import freeze_support
# from turtle import forward
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from numpy import float32
from torch.utils.data import DataLoader
from dataset import ClimateHackDataset
from loss import MS_SSIMLoss
import random
# from submission.model import Model
import numpy as np
# import cv2
from submission.model import *
from pytorch_msssim import MS_SSIM
from einops import rearrange
import pickle
import pytorch_lightning as pl
import torchvision

class PredictionTrainer(pl.LightningModule):
    def __init__(self, config, model=None, device=torch.device('cpu'), convert=False, data_range=1023, channels=24, **args):
        super().__init__()
        self.model = model(config)
        self.criterion = MS_SSIMLoss(channels=channels, data_range=data_range)
        # self.criterion = nn.MSELoss()
        self.config = config 
        self.args = args
        self.convert = convert
        #self.truncated_bptt_steps = 6

    def forward(self, x):
        return self.model(x, **self.args)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])#, weight_decay=1e-8)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor':'valid_loss'}
        # return optimizer

    def training_step(self, training_batch, batch_idx):
        # batch_coordinates, batch_features, batch_targets = training_batch
        batch_features, batch_targets = training_batch[-2:]
        # print(batch_features)
        if self.convert:
            batch_features = rearrange(batch_features, 'b (t c) h w -> b t c h w', c=1) 
        predictions = self.model(batch_features, **self.args)
        if self.convert:
            predictions = rearrange(predictions, 'b t c h w -> b (t c) h w')
        # batch_targets /= self.data_range
        # print(predictions)
        loss = self.criterion(predictions.unsqueeze(dim=2), batch_targets[:,:predictions.shape[1]].unsqueeze(dim=2))
        # print(loss)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, training_batch, batch_idx):
        # batch_coordinates, batch_features, batch_targets = training_batch
        batch_features, batch_targets = training_batch[-2:]
        if self.convert:
            batch_features = rearrange(batch_features, 'b (t c) h w -> b t c h w', c=1) 
        predictions = self.model(batch_features, **self.args)
        if self.convert:
            predictions = rearrange(predictions, 'b t c h w -> b (t c) h w')
        # print('SHAPE', predictions.shape)
        loss = self.criterion(predictions.unsqueeze(dim=2), batch_targets[:,:predictions.shape[1]].unsqueeze(dim=2))
        self.log('valid_loss', loss, prog_bar=True)
        #logging, comment if doesnt work
        # grid = torchvision.utils.make_grid(predictions).view(24, 1, 64, 64)
        # self.logger.experiment.add_images('predictions', grid, 0)

        return loss

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
    #     self.log("ptl/val_loss", avg_loss)

