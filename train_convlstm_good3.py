
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
from einops import rearrange
import pickle
import pytorch_lightning as pl
from training_utils import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from training_config import add_arguments
parser = argparse.ArgumentParser(description='Train skip conn UNet')

add_arguments(parser)

args = vars(parser.parse_args())
print(args)


BATCH_SIZE = 1
EPOCHS = 1
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
N_IMS = 24

if __name__ == '__main__':
    args['innersize'] = list(map(int, args['innersize'].split()))
    # config = {
    #     "lr": args['lr'],
    #     "normalize": args['normalize'], 
    #     "local_norm": args['localnorm'],
    #     "in_opt_flow": False,
    #     "opt_flow": False,
    #     "criterion": args['criterion'],
    #     "output_mean": 0,
    #     "output_std": 0,
    #     "inputs":args['inputs'],
    #     "outputs":args['outputs'],
    #     "dropout": args['dropout'],
    #     "weight_decay":args['weightdecay'],
    #     "inner_size":innertup,
    #     "epochs": args['epochs'],
    #     "dataset": args['dataset'],
    #     "patience": args['patience']
    # }
    train_model(args, ConvLSTM, 'convlstm-3', convert=True)
  
    