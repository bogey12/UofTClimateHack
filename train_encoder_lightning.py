
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
    args['separate'] = list(map(int, args['separate'].split(' ')))
    # config = {
    #     "separate": separate_tup,
    #     "n_layers": args['nlayers'],
    #     "dropout": args['dropout'],
    #     "swap": args['swap'],
    #     "lr": args['lr'],
    #     "normalize": args['normalize'], 
    #     "local_norm": args['localnorm'],
    #     "output_mean": args['outputmean'],
    #     "output_std": args['outputstd'],
    #     "opt_flow":args['optflow'],
    #     "inputs":args['inputs'],
    #     "outputs":args['outputs'],
    #     "in_opt_flow":args['inoptflow'],
    #     "criterion": args['criterion'],
    #     "weight_decay":args['weightdecay'],
    #     "epochs": args['epochs'],
    #     "dataset": args['dataset'],
    #     "patience": args['patience']
    # }
    train_model(args, Encoder, 'conv3d')
    