
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
import argparse

parser = argparse.ArgumentParser(description='Train skip conn UNet')
parser.add_argument('--separate', required=True,
                    help='size', type=str)
parser.add_argument('--nlayers', required=True,
                    help='layers', type=int)
parser.add_argument('--dropout', required=True,
                    help='dropout', type=float)
parser.add_argument('--swap', required=False,
                    help='swap', type=bool, default=False)
parser.add_argument('--lr', required=True,
                    help='lr', type=float)
parser.add_argument('--normalize', required=False,
                    help='norm', type=str, default="standardize")
parser.add_argument('--epochs', required=False,
                    help='epochs', type=int, default=200)
parser.add_argument('--localnorm', required=False,
                    help='local norm', type=bool, default=True)
parser.add_argument('--patience', required=False,
                    help='patience', type=int, default=20)
parser.add_argument('--outputmean', required=False, type=float, default=0)
parser.add_argument('--outputstd', required=False, type=float, default=0)
parser.add_argument('--dataset', required=False, type=str, default="ds-total/ds_total.npz")
parser.add_argument('--optflow', required=False, type=bool, default=False)
parser.add_argument('--inputs', required=False, type=int, default=12)
parser.add_argument('--outputs', required=False, type=int, default=24)
parser.add_argument('--inoptflow', required=False, type=int, default=0)
parser.add_argument('--criterion', required=False, type=str, default="msssim")
args = vars(parser.parse_args())
print(args)


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
    separate_tup = list(map(int, args['separate'].split(' ')))
    # print(separate_tup)
    training_ds = dataset.sel(time=slice("2020-07-01 09:00", "2020-10-01 09:00"))
    validation_ds = dataset.sel(time=slice("2020-12-01 09:00", "2020-12-10 09:00"))
    datapoints = np.load(f'/datastores/{args["dataset"]}', allow_pickle=True)['datapoints']#.to_list()
    # datapoints = np.load('/datastores/data2/data.npz', allow_pickle=True)['data']#.to_list()
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
    # training_dl, validation_dl = [DataLoader(ds, batch_size=BATCH_SIZE, num_workers=1, persistent_workers=True, pin_memory=True) for ds in [ch_training, ch_validation]]
    training_dl, validation_dl = [DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True) for ds in [ch_training, ch_validation]]
    # ch_dataloader = DataLoader(ch_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    # training_dl.multiprocessing_context = 'spawn'   
    # validation_dl.multiprocessing_context = 'spawn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        "separate": separate_tup,
        "n_layers": args['nlayers'],
        "dropout": args['dropout'],
        "swap": args['swap'],
        "lr": args['lr'],
        "normalize": args['normalize'], 
        "local_norm": args['localnorm'],
        "output_mean": args['outputmean'],
        "output_std": args['outputstd'],
        "opt_flow":args['optflow'],
        "inputs":args['inputs'],
        "outputs":args['outputs'],
        "in_opt_flow":args['inoptflow'],
        "criterion": args['criterion']
    }
    training_model = PredictionTrainer(config, model=Encoder, device=device, convert=False)
    early_stop = EarlyStopping('valid_loss', patience=args['patience'], mode='min')
    trainer = pl.Trainer(gpus=1, precision=32, max_epochs=args['epochs'], callbacks=[early_stop], accumulate_grad_batches=7)#, detect_anomaly=True)#, overfit_batches=1)#, benchmark=True)#, limit_train_batches=1)
    trainer.fit(training_model, training_dl, validation_dl)
    torch.save(trainer.model.state_dict(), 'encoder_generic.pt')
    