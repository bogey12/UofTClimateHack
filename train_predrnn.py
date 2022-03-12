
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
from training_config import add_arguments
from submission.predrnn_v2 import *
import os
os.environ["WANDB_CONSOLE"] = "off"

parser = argparse.ArgumentParser(description='Train skip conn UNet')

add_arguments(parser)
args = vars(parser.parse_args())
print(args)

args['numhidden'] = list(map(int, args['numhidden'].split(',')))
def train():
    train_model(args, PredRNNModel, 'predrnn-1')
if args['sweep']:
    wandb.agent(args['sweepid'], function=train, count=args['sweepruns'], entity="loluwot", project="ClimateHack")
else:
    train()
    