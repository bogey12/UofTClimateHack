
import functools
import itertools
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
# from submission.model import *
from models2.forecaster import Forecaster
from models2.encoder import Encoder as Encoder2
from models2.model import EF
from models2.loss import Weighted_mse_mae
# from models2.convLSTM import ConvLSTM as ConvLSTM2
from models2.net_params import return_params
from models2.config import cfg
from pytorch_msssim import MS_SSIM
from einops import rearrange
import pickle
import pytorch_lightning as pl
from training_utils import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from collections import OrderedDict

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
parser.add_argument('--dataset', required=False, type=str, default="/datastores/ds-total/ds_total.npz")
parser.add_argument('--inputs', required=False, type=int, default=12)
parser.add_argument('--outputs', required=False, type=int, default=24)
parser.add_argument('--criterion', required=False, type=str, default="msssim")
parser.add_argument('--weightdecay', required=False,
                    help='lr', type=float, default=1e-8)
parser.add_argument('--innersize', required=False, type=str, default="8 64 192")

args = vars(parser.parse_args())
print(args)

class TempModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        convlstm_encoder_params1, convlstm_forecaster_params1 = return_params(config['inner_size'])
        self.encoder = Encoder2(convlstm_encoder_params1[0], convlstm_encoder_params1[1])
        self.forecaster = Forecaster(convlstm_forecaster_params1[0], convlstm_forecaster_params1[1])
        # self.ef = EF(encoder, forecaster)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, features):
        x = rearrange(features, 'b (c t) h w -> t b c h w', c=1)
        x = self.dropout(x)
        # x = self.ef(x)
        x = self.encoder(x)
        x = self.forecaster(x)
        x = rearrange(x, 't b c h w -> b (t c) h w')
        return x

BATCH_SIZE = 1
EPOCHS = 1
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
N_IMS = 24

if __name__ == '__main__':
    innertup = list(map(int, args['innersize'].split()))
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
        "inner_size":innertup,
        "epochs": args['epochs'],
        "dataset": args['dataset'],
        "patience": args['patience']
    }
    train_model(config, TempModel, 'convlstm-2')
  
    