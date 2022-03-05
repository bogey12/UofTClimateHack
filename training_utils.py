
from audioop import avg
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from collections import defaultdict
import wandb


BATCH_SIZE = 1
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
class PredictionTrainer(pl.LightningModule):
    def __init__(self, config, model=None, device=None, convert=False, data_range=1023, **args):
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
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.7)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.config['lr'], max_lr=0.1, cycle_momentum=False, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor':'valid_loss'}
        # return optimizer

    def preprocessing(self, batch_features):
        MEAN = 299.17117
        STD = 146.06215
        mi1 = None
        ma1 = None
        if self.config['in_opt_flow']:
            batch_features = rearrange(batch_features, 'b t h w c -> b (t c) h w', c=2) 
        # print(batch_features)
        if self.convert:
            batch_features = rearrange(batch_features, 'b (t c) h w -> b t c h w', c=1)
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
        batch_features, extra = self.preprocessing(batch_features)
        predictions = self.model(batch_features, **self.args)
        if self.convert:
            predictions = rearrange(predictions, 'b t c h w -> b (t c) h w')
        predictions = self.postprocessing(predictions, extra)
        # if self.config['opt_flow']:
        #     predictions = rearrange(predictions, 'b (t c) h w -> b t h w c', c=2)
        # target = torch.tensor(batch_targets).view(1, 24, 64, 64)
        if not self.config['opt_flow']:
            loss = self.criterion(predictions.unsqueeze(dim=2), batch_targets[:,:self.config['outputs']].unsqueeze(dim=2))
        else:
            predictions = rearrange(predictions, 'b (t c) h w -> b t h w c', c=2)
            # target = rearrange(batch_targets[:,:predictions.shape[1]], 'b t h w c -> b (t c) h w')
            loss = self.criterion(predictions, batch_targets[:,:predictions.shape[1]])

        self.log('valid_loss', loss, prog_bar=True)
        #logging, comment if doesnt work
        grid_expected = wandb.Image(torchvision.utils.make_grid([batch_targets[:, i] for i in range(self.config['outputs'])]))
        grid_predicted = wandb.Image(torchvision.utils.make_grid([predictions[:, i] for i in range(self.config['outputs'])]))
        wandb.log({"predictions":grid_predicted, "expected": grid_expected})
        # self.logger.experiment.add_images('predictions', grid, 0)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        # self.log("ptl/val_loss", avg_loss)
        wandb.log({'avg_loss':avg_loss})

def train_model(config, model_class, name, convert=False, **args):
    #add dataset to config  
    #add epochs to config  
    #add name to config
    #add patience to config
    wandb.init(project="ClimateHack", entity="loluwot")
    wandb_logger = WandbLogger(project="ClimateHack")
    dataset = xr.open_dataset(
        SATELLITE_ZARR_PATH, 
        engine="zarr",
        chunks="auto",  # Load the data as a Dask array
    )
    # print(separate_tup)
    training_ds = dataset.sel(time=slice("2020-07-01 09:00", "2020-10-01 09:00"))
    validation_ds = dataset.sel(time=slice("2020-12-01 09:00", "2020-12-10 09:00"))
    datapoints = np.load(f'{config["dataset"]}', allow_pickle=True)['datapoints']#.to_list()
    valid_datapoints = np.load(f'{"valid_set.npz" if "validation" not in config else config["validation"]}', allow_pickle=True)['datapoints']#.to_list()
    np.random.shuffle(datapoints)
    tot_points = len(datapoints) 
    validation_size = len(valid_datapoints)
    datapoints = datapoints.reshape((tot_points, 2))
    valid_datapoints = valid_datapoints.reshape((validation_size, 2))
    training = datapoints.tolist()
    testing = valid_datapoints.tolist()

    ch_training = ClimateHackDataset(training_ds, crops_per_slice=5, day_limit=7, outputs=config['outputs'])#, timeskip=8)
    ch_training.cached_items = training# + testing
    ch_validation = ClimateHackDataset(validation_ds, crops_per_slice=5, day_limit=3, outputs=config['outputs'])#, cache=False)
    ch_validation.cached_items = testing
    training_dl, validation_dl = [DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True) for ds in [ch_training, ch_validation]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="submission/",
        filename= name + "-{epoch:02d}-{valid_loss:.4f}",
        save_top_k=3,
        mode="min",
        save_weights_only=True
    )
    training_model = PredictionTrainer(config, model=model_class, device=device, convert=convert)
    early_stop = EarlyStopping('valid_loss', patience=config['patience'], mode='min')
    trainer = pl.Trainer(gpus=1, precision=32, max_epochs=config['epochs'], callbacks=[early_stop, checkpoint_callback], accumulate_grad_batches=7, gradient_clip_val=50.0, logger=wandb_logger, **args)#, detect_anomaly=True)#, overfit_batches=1)#, benchmark=True)#, limit_train_batches=1)
    trainer.fit(training_model, training_dl, validation_dl)
    torch.save(trainer.model.state_dict(), f'{name}.pt')