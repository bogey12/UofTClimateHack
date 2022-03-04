
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import argparse
import wandb
wandb.init(project="ClimateHack", entity="loluwot")
wandb_logger = WandbLogger(project="ClimateHack")

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
parser.add_argument('--weightdecay', required=False, type=float, default=1e-8)
args = vars(parser.parse_args())
print(args)


BATCH_SIZE = 1
EPOCHS = 1
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
N_IMS = 24

if __name__ == '__main__':
    separate_tup = list(map(int, args['separate'].split(' ')))
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
        "criterion": args['criterion'],
        "weight_decay":args['weightdecay'],
        "epochs": args['epochs'],
        "dataset": args['dataset'],
        "patience": args['patience']
    }
    train_model(config, Encoder, 'conv3d', logger=wandb_logger)
    