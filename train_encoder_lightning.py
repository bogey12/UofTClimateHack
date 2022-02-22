
from multiprocessing.dummy import freeze_support
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from numpy import float32
from torch.utils.data import DataLoader, random_split
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
from training_utils import *
from pytorch_lightning.callbacks import EarlyStopping
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback



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
    validation_ds = dataset.sel(time=slice("2020-12-01 09:00", "2020-12-10 09:00"))
    datapoints = np.load('~/datastores/ds_total/ds_total.npz', allow_pickle=True)['datapoints']#.to_list()
    np.random.shuffle(datapoints)
    tot_points = len(datapoints)
    train_len = int(tot_points*0.8)
    datapoints = datapoints.reshape((tot_points, 2))
    training = datapoints[:train_len].tolist()
    testing = datapoints[train_len:].tolist()

    ch_training = ClimateHackDataset(training_ds, crops_per_slice=5, day_limit=7, outputs=N_IMS, shuffler=False)#, timeskip=8)
    ch_training.cached_items = training# + testing
    ch_validation = ClimateHackDataset(validation_ds, crops_per_slice=5, day_limit=3, outputs=N_IMS)#, cache=False)
    ch_validation.cached_items = testing
    training_dl, validation_dl = [DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1, persistent_workers=True, pin_memory=True) for ds in [ch_training, ch_validation]]
    # training_dl, validation_dl = [DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True) for ds in [ch_training, ch_validation]]
    # ch_dataloader = DataLoader(ch_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    training_dl.multiprocessing_context = 'spawn'   
    validation_dl.multiprocessing_context = 'spawn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "separate": (128, False),
        "n_layers": 1,
        "dropout": 0.1,
        "swap": False,
        "lr": 0.01
    }
    training_model = PredictionTrainer(config, model=Encoder, device=device, convert=False)
    early_stop = EarlyStopping('valid_loss', patience=30, mode='min')
    trainer = pl.Trainer(gpus=1, precision=32, max_epochs=200,callbacks=[early_stop], accumulate_grad_batches=7)#, detect_anomaly=True)#, overfit_batches=1)#, benchmark=True)#, limit_train_batches=1)
    trainer.fit(training_model, training_dl, validation_dl)
    # torch.save(trainer.model.state_dict(), 'submission/encoder_generic.pt')
    