
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
from pytorch_lightning.callbacks import EarlyStopping


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
    datapoints = np.load('ds_total.npz', allow_pickle=True)['datapoints']#.to_list()
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
    # encoder = [('conv', 'leaky', 1, 64, 3, 1, 2),
    #          ('convlstm', '', 64, 64, 3, 1, 1),
    #          ('conv', 'leaky', 64, 64, 3, 1, 2),
    #          ('convlstm', '', 64, 64, 3, 1, 1),
    #          ('conv', 'leaky', 64, 128, 3, 1, 2),
    #          ('convlstm', '', 128, 128, 3, 1, 1)]
    # decoder = [('deconv', 'leaky', 128, 64, 4, 1, 2),
    #         ('convlstm', '', 128, 64, 3, 1, 1),
    #         ('deconv', 'leaky', 64, 64, 4, 1, 2),
    #         ('convlstm', '', 128, 64, 3, 1, 1),
    #         ('deconv', 'leaky', 64, 64, 4, 1, 2),
    #         ('convlstm', '', 65, 64, 3, 1, 1),
    #         ('conv_output', '', 64, 1, 1, 0, 2)]

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
               ('conv_output', '', 32, 1, 1, 0, 2)]
    # encoder = [('conv', 'leaky', 1, 64, 7, 3, 2),
    #          ('convlstm', '', 64, 64, 7, 3, 1),
    #          ('conv', 'leaky', 64, 64, 7, 3, 2),
    #          ('convlstm', '', 64, 64, 7, 3, 1),]
    # decoder = [('deconv', 'leaky', 64, 64, 8, 3, 2),
    #             ('convlstm', '', 128, 64, 7, 3, 1),
    #             ('deconv', 'leaky', 64, 64, 8, 3, 2),
    #             ('convlstm', '', 65, 64, 7, 3, 1),
    #             ('conv', 'sigmoid', 64, 1, 1, 0, 2)]
    # encoder = [('conv', 'leaky', 1, 64, 5, 2, 2),
    #          ('convlstm', '', 64, 64, 3, 1, 1),
    #          ('conv', 'leaky', 64, 64, 3, 1, 2),
    #          ('convlstm', '', 64, 64, 3, 1, 1),]
    # decoder = [('deconv', 'leaky', 64, 64, 4, 1, 2),
    #             ('convlstm', '', 128, 64, 3, 1, 1),
    #             ('deconv', 'leaky', 64, 64, 4, 1, 2),
    #             ('convlstm', '', 65, 64, 3, 1, 1),
    #             ('conv', 'relu', 64, 1, 1, 0, 2)]
    # encoder = [('conv', 'leaky', 1, 128, 7, 3, 2),
    #          ('convlstm', '', 128, 64, 7, 3, 1),
    #          ('conv', 'leaky', 64, 64, 7, 3, 2),
    #          ('convlstm', '', 64, 64, 7, 3, 1),]
    # decoder = [('deconv', 'leaky', 64, 64, 8, 3, 2),
    #         ('convlstm', '', 128, 64, 7, 3, 1),
    #         ('deconv', 'leaky', 64, 64, 8, 3, 2),
    #         ('convlstm', '', 65, 64, 7, 3, 1),  
    #         ('conv', 'relu', 64, 1, 1, 0, 1)]
    # encoder = [('conv', 'leaky', 1, 128, 5, 2, 2),
    #          ('convlstm', '', 128, 128, 5, 2, 1)]
    # decoder = [('deconv', 'leaky', 128, 128, 6, 2, 2),
    #             ('convlstm', '', 129, 128, 5, 2, 1),
    #             ('conv', 'relu', 128, 1, 1, 0, 2)]
    # encoder = [('conv', 'leaky', 1, 128, 5, 2, 2),
    #          ('convlstm', '', 128, 64, 5, 2, 1)]
    # decoder = [('deconv', 'leaky', 64, 64, 6, 2, 2),
    #             ('convlstm', '', 65, 64, 5, 2, 1),
    #             ('conv_output', '', 64, 1, 1, 0, 2)]
    model = ConvLSTM(encoder, decoder, dropout=0.2)
    # profiler = PyTorchProfiler()
    early_stop = EarlyStopping('valid_loss', patience=30, mode='min')
    training_model = PredictionTrainer(model=model, device=device, convert=True, data_range=1023, channels=12, fac=1023)
    # training_model.load_state_dict(torch.load('submission/convlstm.pt'))

    trainer = pl.Trainer(gpus=1, precision=32, max_epochs=200, callbacks=[early_stop])#, overfit_batches=1)#, benchmark=True)#, limit_train_batches=1)#, detect_anomaly=True)
    trainer.fit(training_model, training_dl, validation_dl)
    torch.save(trainer.model.state_dict(), f'submission/convlstm.pt')