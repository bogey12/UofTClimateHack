
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
import math
import torch.nn.functional as F
from training_config import *
from processing_utils import *
from string import ascii_lowercase, digits
WANDB_DISABLED = False
# BATCH_SIZE = 1
NUM_IMAGES = 10
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
class PredictionTrainer(pl.LightningModule):
    def __init__(self, config, model=None, device=None, convert=False, data_range=1023, **args):
        super().__init__()
        config['device'] = device
        self.model = model(config)
        self.model = self.model.to(device)
        if config['criterion'] == 'mse':
            self.criterion = nn.MSELoss()
        elif config['criterion'] == 'msssim':
            channels = config['outputs']
            if config['model_name'] == 'predrnn':
                channels += config['inputs'] - 1
                self.eval_criterion = MS_SSIMLoss(channels=config['outputs'], data_range=data_range)
            self.criterion = MS_SSIMLoss(channels=channels, data_range=data_range)
        
        # self.criterion = nn.MSELoss()
        self.config = config 
        # self.config['device'] = device
        self.args = args
        self.convert = convert
        self.logged = sorted(random.sample(list(range(0, 200//self.config['batch_size'])), k=10))
        self.lr = None
        self.downconv = nn.Identity()
        if config['downsample']:
            if config['downsample'] == 'stride':
                self.downconv = nn.Conv2d(config['inputs'], config['inputs'], 3, stride=2, padding=1) #downsample by 1/2
            elif config['downsample'] == 'maxpool':
                self.downconv = nn.MaxPool2d(2)
        #self.truncated_bptt_steps = 6

    def forward(self, x):
        y, extra = preprocessing(self.config, x)
        # print(x.shape)
        x = self.model(y, **self.args)
        if self.config['model_name'] == 'predrnn':
            x, loss = x
        x = postprocessing(self.config, x, extra)
        if self.config['model_name'] == 'predrnn':
            return x, loss
        return x

    def configure_optimizers(self):
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=(self.lr or self.config['lr']), weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=(self.lr or self.config['lr']), weight_decay=self.config['weight_decay'], momentum=self.config['momentum'])
        
        if self.config['lr_scheduler'] == 'plateau':
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.config['scheduler_patience'], verbose=True, factor=self.config['scheduler_gamma'])
        elif self.config['lr_scheduler'] == 'cyclic':
            lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.config['lr'], max_lr=self.config['scheduler_max'], cycle_momentum=False, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor':'valid_loss'}
        # return optimizer

    def training_step(self, training_batch, batch_idx):
        # batch_coordinates, batch_features, batch_targets = training_batch
        batch_features, batch_targets = training_batch[-2:]
        features = batch_features
        features = self.downconv(features)
        if self.config['model_name'] == 'predrnn':
            features = torch.cat((features, batch_targets), dim=1).clone() #b (c t) h w
        
        predictions = self.forward(features, **self.args)
        # print(predictions)
        if self.config['opt_flow']:
            predictions = rearrange(predictions, 'b (t c) h w -> b t h w c', c=2)
            # target = rearrange(batch_targets[:,:predictions.shape[1]], 'b t h w c -> b (t c) h w')
            loss = self.criterion(predictions, batch_targets[:,:predictions.shape[1]])
        
        elif self.config['model_name'] == 'predrnn':
            predictions, decouple_loss = predictions
            # print('PREDICTIONS:', predictions.unsqueeze(dim=2).shape)
            net_input = features.unsqueeze(dim=2)[:, 1:]
            # print('NETINPUT:', net_input.shape)
            loss = decouple_loss + self.criterion(predictions.unsqueeze(dim=2), net_input)
            if self.config['reverse_input']:
                flipped_features = torch.flip(features, [1])
                net_input_f = flipped_features.unsqueeze(dim=2)[:, 1:]
                predictions2, decouple_loss2 = self.forward(flipped_features, **self.args)
                loss += decouple_loss2 + self.criterion(predictions2.unsqueeze(dim=2), net_input_f)
                loss /= 2
        else:
            loss = self.criterion(predictions.unsqueeze(dim=2), batch_targets[:,:24].unsqueeze(dim=2))

        # loss = self.criterion(predictions, batch_targets[:,:predictions.shape[1]])
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('eta', self.model.eta, prog_bar=True, sync_dist=True)
        wandb.log({'eta': self.model.eta})
        return loss


    def validation_step(self, training_batch, batch_idx):
        # batch_coordinates, batch_features, batch_targets = training_batch
        batch_features, batch_targets = training_batch[-2:]
        features = batch_features
        features = self.downconv(features)
        if self.config['model_name'] == 'predrnn':
            features = torch.cat((features, batch_targets), dim=1) #b (c t) h w
        predictions = self.forward(features, **self.args)
        # print(predictions)
        if self.config['opt_flow']:
            predictions = rearrange(predictions, 'b (t c) h w -> b t h w c', c=2)
            # target = rearrange(batch_targets[:,:predictions.shape[1]], 'b t h w c -> b (t c) h w')
            loss = self.criterion(predictions, batch_targets[:,:predictions.shape[1]])
        
        elif self.config['model_name'] == 'predrnn':
            predictions, decouple_loss = predictions
            easy_predictions, _ = self.forward(features, override=True)
            # print('PREDICTIONS:', predictions.unsqueeze(dim=2).shape)
            expected = batch_targets.unsqueeze(dim=2)
            # print('NETINPUT:', net_input.shape)
            loss = self.eval_criterion(predictions.unsqueeze(dim=2)[:, -self.config['outputs']:], expected)
        else:
            loss = self.criterion(predictions.unsqueeze(dim=2), batch_targets[:,:24].unsqueeze(dim=2))

        if len(self.logged) > 0 and batch_idx == self.logged[0]:
            grid_expected = wandb.Image(torchvision.utils.make_grid([batch_targets[:1, i] for i in range(self.config['outputs'])]))
            if self.config['model_name'] == 'predrnn':
                grid_predicted = wandb.Image(torchvision.utils.make_grid([easy_predictions[:1, i + self.config['inputs'] - 1] for i in range(self.config['outputs'])]))
            else:
                grid_predicted = wandb.Image(torchvision.utils.make_grid([predictions[:1, i] for i in range(self.config['outputs'])]))

            wandb.log({"predictions":grid_predicted, "expected": grid_expected})
            self.logged.pop(0)

        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        wandb.log({'valid_loss':loss})
        return loss

    def validation_epoch_end(self, outputs):
        if len(outputs) > 0:  
            # print(outputs)      
            avg_loss = torch.stack(outputs).mean()
            # print(avg_loss)
            self.log('avg_loss', avg_loss)
            wandb.log({'avg_loss':avg_loss})
        self.logged = sorted(random.sample(list(range(0, 200//self.config['batch_size'])), k=10))


def train_model(rawargs, model_class, name, **args):
    #add dataset to config  
    #add epochs to config  
    #add name to config
    #add patience to config

    rawargs = defaultdict(lambda: None, rawargs)
    config = dict([(k, rawargs[k.replace('_', '')] if rawargs[k.replace('_', '')] is not None else v) for k, v in default_config.items()])
    random_str = ''.join(random.choices(ascii_lowercase + digits, k=5))

    if config['sweep']:
        # wandb.init(config=config, project="ClimateHack", entity="loluwot")
        wandb.init(config=config, project="ClimateHack", entity="loluwot", mode='disabled' if WANDB_DISABLED else None)
    else:
        # wandb.init(config=config, project="ClimateHack", entity="loluwot", name=f'{name}-{random_str}', group=name)
        wandb.init(config=config, project="ClimateHack", entity="loluwot", name=f'{name}-{random_str}', group=name,  mode='disabled' if WANDB_DISABLED else None)

    config = {**config, **wandb.config}

    wandb_logger = True# if config['sweep'] else WandbLogger(project="ClimateHack")
    dataset = xr.open_dataset(
        SATELLITE_ZARR_PATH, 
        engine="zarr",
        chunks="auto",  # Load the data as a Dask array
    )
    # print(separate_tup)
    training_ds = dataset.sel(time=slice("2020-07-01 09:00", "2020-10-01 09:00"))
    validation_ds = dataset.sel(time=slice("2020-12-01 09:00", "2020-12-10 09:00"))
    loaded_file = np.load(f'{config["dataset"]}', allow_pickle=True)
    datapoints = loaded_file['datapoints']#.to_list()
    
    if config['internal_valid']:
        valid_datapoints = loaded_file['valid']
    else:
        valid_datapoints = np.load(f'{config["validation"] or "valid_set.npz"}', allow_pickle=True)['datapoints']#.to_list()
    
    np.random.shuffle(datapoints)
    tot_points = len(datapoints) 
    validation_size = len(valid_datapoints)
    datapoints = datapoints.reshape((tot_points, 2))
    sl = config['dataset_slice']
    if sl < 0:
        sl += tot_points + 1
    datapoints = datapoints[:sl]
    valid_datapoints = valid_datapoints.reshape((validation_size, 2))
    training = datapoints.tolist()
    testing = valid_datapoints.tolist()

    ch_training = ClimateHackDataset(training_ds, crops_per_slice=5, day_limit=7, outputs=config['outputs'])#, timeskip=8)
    ch_training.cached_items = training# + testing
    ch_validation = ClimateHackDataset(validation_ds, crops_per_slice=5, day_limit=3, outputs=config['outputs'])#, cache=False)
    ch_validation.cached_items = testing
    training_dl, validation_dl = [DataLoader(ds, batch_size=config['batch_size'], num_workers=0, pin_memory=True, drop_last=True) for ds in [ch_training, ch_validation]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_callback = ModelCheckpoint(
        monitor = "avg_loss",
        dirpath = "submission/",
        filename = name + "-{epoch:02d}-{avg_loss:.4f}",
        save_top_k=3,
        mode="min",
        save_weights_only=True
    )
    if config['checkpoint'] != '':
        training_model = PredictionTrainer.load_from_checkpoint(config['checkpoint'], config=config, model=model_class, device=device)
    else:
        training_model = PredictionTrainer(config, model=model_class, device=device)
    early_stop = EarlyStopping('valid_loss', patience=config['patience'], mode='min')
    if config['gpu'] != 1:
        args['strategy'] = "ddp"#, detect_anomaly=True)#, overfit_batches=1)#, benchmark=True)#, limit_train_batches=1)
    
    trainer = pl.Trainer(gpus=config['gpu'], precision=32, max_epochs=config['epochs'], callbacks=[early_stop, checkpoint_callback], accumulate_grad_batches=max(1, config['accumulate']//config['batch_size']), gradient_clip_val=50.0, logger=wandb_logger, **args)#, detect_anomaly=True)#, overfit_batches=1)#, benchmark=True)#, limit_train_batches=1)
    trainer.fit(training_model, training_dl, validation_dl)
    torch.save(trainer.model.state_dict(), f'{name}.pt')

def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
    
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    
    return patches.view(b,-1,patches.shape[-2], patches.shape[-1])