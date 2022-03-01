
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
from models2.loss import Weighted_mse_mae


class PredictionTrainer(pl.LightningModule):
    def __init__(self, config, model=None, device=torch.device('cpu'), convert=False, data_range=1023, **args):
        super().__init__()
        self.model = model(config)
        if config['criterion'] == 'mse':
            self.criterion = nn.MSELoss()
        elif config['criterion'] == 'msssim':
            self.criterion = MS_SSIMLoss(channels=config['outputs'], data_range=data_range)

        # self.criterion = nn.MSELoss()
        self.config = config 
        self.args = args
        self.convert = convert
        #self.truncated_bptt_steps = 6

    def forward(self, x):
        return self.model(x, **self.args)

    def configure_optimizers(self):
        decay = 0 if 'weight_decay' not in self.config else self.config['weight_decay']
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.7)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor':'valid_loss'}
        # return optimizer

    def preprocessing(self, batch_features):
        MEAN = 299.17117
        STD = 146.06215
        mi1 = None
        ma1 = None
        if self.config['normalize']:
            if self.config['local_norm']:
                MEAN = torch.mean(batch_features)
                STD = torch.std(batch_features)
            if self.config['normalize'] == 'standardize':
                batch_features -= MEAN
                batch_features /= STD
            elif self.config['normalize'] == 'minmax':
                mi1 = batch_features.min()
                ma1 = batch_features.max()
                batch_features -= mi1
                batch_features /= (ma1 - mi1)
                batch_features *= 2
                batch_features -= 1
        return batch_features, {'MEAN':MEAN, 'STD':STD, 'mi1':mi1, 'ma1':ma1}

    def postprocessing(self, predictions, extra):
        MEAN = extra['MEAN']
        STD = extra['STD']
        mi1 = extra['mi1']
        ma1 = extra['ma1']
        if self.convert:
            predictions = rearrange(predictions, 'b t c h w -> b (t c) h w')
        if self.config['normalize']:
            if self.config['output_std']:
                MEAN = self.config['output_mean']
                STD = self.config['output_std']
            if self.config['normalize'] == 'standardize':
                predictions *= STD
                predictions += MEAN
            elif self.config['normalize'] == 'minmax':
                predictions += 1
                predictions /= 2
                predictions *= (ma1 - mi1)
                predictions += mi1
            # predictions *= 1023
        return predictions

    def training_step(self, training_batch, batch_idx):
        # batch_coordinates, batch_features, batch_targets = training_batch
        batch_features, batch_targets = training_batch[-2:]
        if self.config['in_opt_flow']:
            batch_features = rearrange(batch_features, 'b t h w c -> b (t c) h w', c=2) 
        # print(batch_features)
        if self.convert:
            batch_features = rearrange(batch_features, 'b (t c) h w -> b t c h w', c=1)
        batch_features, extra = self.preprocessing(batch_features)
        predictions = self.model(batch_features, **self.args)
        predictions = self.postprocessing(predictions, extra)
        # if self.config['optflow']:
        #     predictions = rearrange(predictions, 'b (t c) h w -> b t h w c', c=2)
        # batch_targets /= self.data_range
        # print(predictions)
        if not self.config['opt_flow']:
            loss = self.criterion(predictions.unsqueeze(dim=2), batch_targets[:,:24].unsqueeze(dim=2))
        else:
            predictions = rearrange(predictions, 'b (t c) h w -> b t h w c', c=2)
            # target = rearrange(batch_targets[:,:predictions.shape[1]], 'b t h w c -> b (t c) h w')
            loss = self.criterion(predictions, batch_targets[:,:predictions.shape[1]])
        # loss = self.criterion(predictions, batch_targets[:,:predictions.shape[1]])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, training_batch, batch_idx):
        # batch_coordinates, batch_features, batch_targets = training_batch
        batch_features, batch_targets = training_batch[-2:]
        if self.config['in_opt_flow']:
            batch_features = rearrange(batch_features, 'b t h w c -> b (t c) h w', c=2) 
        if self.convert:
            batch_features = rearrange(batch_features, 'b (t c) h w -> b t c h w', c=1) 
        batch_features, extra = self.preprocessing(batch_features)
        predictions = self.model(batch_features, **self.args)
        if self.convert:
            predictions = rearrange(predictions, 'b t c h w -> b (t c) h w')
        predictions = self.postprocessing(predictions, extra)
        # if self.config['opt_flow']:
        #     predictions = rearrange(predictions, 'b (t c) h w -> b t h w c', c=2)
        # target = torch.tensor(batch_targets).view(1, 24, 64, 64)
        if not self.config['opt_flow']:
            loss = self.criterion(predictions.unsqueeze(dim=2), batch_targets[:,:24].unsqueeze(dim=2))
        else:
            predictions = rearrange(predictions, 'b (t c) h w -> b t h w c', c=2)
            # target = rearrange(batch_targets[:,:predictions.shape[1]], 'b t h w c -> b (t c) h w')
            loss = self.criterion(predictions, batch_targets[:,:predictions.shape[1]])

        self.log('valid_loss', loss, prog_bar=True)
        #logging, comment if doesnt work
        # grid = torchvision.utils.make_grid(predictions).view(24, 1, 64, 64)
        # self.logger.experiment.add_images('predictions', grid, 0)

        return loss

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
    #     self.log("ptl/val_loss", avg_loss)

