
from multiprocessing.dummy import freeze_support
from turtle import forward
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from numpy import float32
from torch.utils.data import DataLoader, random_split
from dataset import ClimateHackDataset
from loss import MS_SSIMLoss
from pytorch_lightning.profiler import PyTorchProfiler
import random
# from submission.model import Model
import numpy as np
# import cv2
from submission.model import *
from pytorch_msssim import MS_SSIM
from einops import rearrange
import pickle
import pytorch_lightning as pl
from training_utils import *
import random
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import argparse
parser = argparse.ArgumentParser(description='Train skip conn UNet')
parser.add_argument('--lr', required=True,
                    help='lr', type=float)
parser.add_argument('--dropout', required=False,
                    help='lr', type=float, default=0.1)
parser.add_argument('--normalize', required=False,
                    help='norm', type=str, default="standardize")
parser.add_argument('--epochs', required=False,
                    help='epochs', type=int, default=200)
parser.add_argument('--localnorm', required=False,
                    help='local norm', type=bool, default=True)
parser.add_argument('--patience', required=False,
                    help='patience', type=int, default=20)
parser.add_argument('--dataset', required=False, type=str, default="ds-total/ds_total.npz")
parser.add_argument('--inputs', required=False, type=int, default=12)
parser.add_argument('--outputs', required=False, type=int, default=24)
parser.add_argument('--criterion', required=False, type=str, default="msssim")
parser.add_argument('--weightdecay', required=False,
                    help='lr', type=float, default=1e-8)

args = vars(parser.parse_args())

BATCH_SIZE = 1
EPOCHS = 1
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
N_IMS = 24

if __name__ == '__main__':
    dataset = xr.open_dataset(
        SATELLITE_ZARR_PATH, 
        engine="zarr",
        chunks="auto",  # Load the data as a Dask array
    )
    training_ds = dataset.sel(time=slice("2020-07-01 09:00", "2020-10-01 09:00"))
    validation_ds = dataset.sel(time=slice("2020-10-01 09:00", "2020-10-10 09:00"))
    datapoints = np.load(f'/datastores/{args["dataset"]}', allow_pickle=True)['datapoints']#.to_list()
    np.random.shuffle(datapoints)
    tot_points = len(datapoints)
    train_len = int(tot_points*0.8)
    datapoints = datapoints.reshape((tot_points, 2))
    training = datapoints[:train_len].tolist()
    testing = datapoints[train_len:].tolist()

    ch_training = ClimateHackDataset(training_ds, crops_per_slice=10, day_limit=7, outputs=N_IMS, timeskip=8)#, shuffler=False)
    ch_training.cached_items = training
    ch_validation = ClimateHackDataset(validation_ds, crops_per_slice=10, day_limit=3, outputs=N_IMS)#, shuffler=False)
    ch_validation.cached_items = testing
    # training_dl, validation_dl = [DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0,  pin_memory=True, persistent_workers=True) for ds in [ch_training, ch_validation]]
    training_dl, validation_dl = [DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True) for ds in [ch_training, ch_validation]]
    # training_dl.multiprocessing_context = 'spawn'
    # validation_dl.multiprocessing_context = 'spawn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = [('conv', 'leaky', 1, 32, 5, 2, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1),
             ('conv', 'leaky', 64, 128, 3, 1, 2),
             ('convlstm', '', 128, 128, 3, 1, 1)]
    decoder = [('deconv', 'leaky', 128, 64, 4, 1, 2),
               ('convlstm', '', 128, 64, 3, 1, 1),
               ('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 32, 4, 1, 2),
               ('convlstm', '', 33, 32, 3, 1, 1),
               ('convoutput', '', 32, 1, 1, 0, 2)]

    config = {
        "lr": args['lr'],
        "normalize": args['normalize'], 
        "local_norm": args['localnorm'],
        "in_opt_flow": False,
        "opt_flow": False,
        "criterion": args['criterion'],
        "output_mean": 0,
        "output_std": 0,
        "inputs":args['inputs'],
        "outputs":args['outputs'],
        "dropout": args['dropout'],
        "weight_decay":args['weightdecay'],
        "encoder": encoder,
        "decoder": decoder
    }

    # profiler = PyTorchProfiler()
    early_stop = EarlyStopping('valid_loss', patience=args['patience'], mode='min')
    training_model = PredictionTrainer(config, model=ConvLSTM, device=device, convert=True, data_range=1023)
    # training_model.load_state_dict(torch.load('submission/convlstm.pt'))
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="submission/",
        filename="convlstm-{epoch:02d}-{valid_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    trainer = pl.Trainer(gpus=1, precision=32, max_epochs=args['epochs'], callbacks=[early_stop, checkpoint_callback], accumulate_grad_batches=7)#, detect_anomaly=True)#, overfit_batches=1)#, benchmark=True)#, limit_train_batches=1)
    trainer.fit(training_model, training_dl, validation_dl)
    torch.save(trainer.model.state_dict(), f'submission/convlstm.pt')