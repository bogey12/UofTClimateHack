
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

BATCH_SIZE = 1
EPOCHS = 1
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
N_IMS = 24


def reserve_schedule_sampling_exp(itr):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (args.batch_size, args.input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - 2,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return real_input_flag


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


if __name__ == '__main__':
    dataset = xr.open_dataset(
        SATELLITE_ZARR_PATH, 
        engine="zarr",
        chunks="auto",  # Load the data as a Dask array
    )
    training_ds = dataset.sel(time=slice("2020-07-01 09:00", "2020-10-01 09:00"))
    validation_ds = dataset.sel(time=slice("2020-10-01 09:00", "2020-10-10 09:00"))
    ch_training = ClimateHackDataset(training_ds, crops_per_slice=10, day_limit=7, outputs=N_IMS, timeskip=8)
    ch_validation = ClimateHackDataset(validation_ds, crops_per_slice=10, day_limit=3, outputs=N_IMS)
    training_dl, validation_dl = [DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True) for ds in [ch_training, ch_validation]]
    # ch_dataloader = DataLoader(ch_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    training_dl.multiprocessing_context = 'spawn'   
    validation_dl.multiprocessing_context = 'spawn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(inputs=12, outputs=N_IMS, fac=1023, half_size=True)
    training_model = PredictionTrainer(model=model, device=device, convert=False)
    trainer = pl.Trainer(gpus=1, precision=64)#, benchmark=True)#, limit_train_batches=1)
    trainer.fit(training_model, training_dl, validation_dl)