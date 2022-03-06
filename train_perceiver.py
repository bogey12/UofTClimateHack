import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from numpy import float32
from torch.utils.data import DataLoader, random_split
from dataset import ClimateHackDataset, ClimateHackDataset2
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
from transformers import PerceiverModel, PerceiverConfig, PerceiverForOpticalFlow
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor, PerceiverOpticalFlowDecoder

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
        fourier_position_encoding_kwargs_preprocessor = dict(
            num_bands=64,
            max_resolution=(128, 128),
            # max_resolution=(128, 496),
            sine_only=False,
            concat_pos=True,
        )
        fourier_position_encoding_kwargs_decoder = dict(
            concat_pos=True, max_resolution=(128, 128), num_bands=64, sine_only=False
        )
        pconfig = PerceiverConfig(use_query_residual=True, 
                                d_model=322, 
                                train_size=(128,128),
                                num_cross_attention_heads=1)

        image_preprocessor = PerceiverImagePreprocessor(
            pconfig,
            prep_type="patches",
            spatial_downsample=1,
            in_channels=1,
            conv_after_patching=True,
            conv_after_patching_in_channels=12*9,
            temporal_downsample=12,
            position_encoding_type="fourier",
            # position_encoding_kwargs
            fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
        )
        self.perceiver = PerceiverModel(
            pconfig,
            input_preprocessor=image_preprocessor,
            decoder=PerceiverOpticalFlowDecoder(
                pconfig,
                num_channels=image_preprocessor.num_channels,
                output_image_shape=(128, 128),
                rescale_factor=100.0,
                use_query_residual=False,
                output_num_channels=24,
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_decoder,
            ),
        )
        # self.ef = EF(encoder, forecaster)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, features):
        x = rearrange(features, 'b (t c) h w -> (b t) c h w', c=1)
        x = extract_image_patches(x, 3).view(1, 12, 9, 128, 128)
        x = self.dropout(x)
        # x = self.ef(x)
        x = self.perceiver(x).logits
        x = rearrange(x, 'b h w t -> b t h w')
        return x[:, :, 32:96, 32:96]

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
    train_model(config, TempModel, 'perceiver')
  
    