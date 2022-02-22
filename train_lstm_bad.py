from multiprocessing.dummy import freeze_support
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from numpy import float32
from torch.utils.data import DataLoader
from dataset import ClimateHackDataset
from loss import MS_SSIMLoss
import random
# from submission.model import Model
import numpy as np
# import cv2
from submission.model import *
from pytorch_msssim import MS_SSIM
import pickle
from einops import rearrange
# plt.rcParams["figure.figsize"] = (20, 12)
BATCH_SIZE = 1
EPOCHS = 10
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
N_IMS = 24
# print(dataset)


if __name__ == '__main__':
    # freeze_support()
    dataset = xr.open_dataset(
        SATELLITE_ZARR_PATH, 
        engine="zarr",
        chunks="auto",  # Load the data as a Dask array
    )
    ch_dataset = ClimateHackDataset(dataset, crops_per_slice=5, day_limit=7, outputs=N_IMS, timeskip=10)
    datapoints = pickle.load(open('datapoints.pkl', 'rb'))
    ch_dataset.cached_items = datapoints

    ch_dataloader = DataLoader(ch_dataset,  batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    ch_dataloader.multiprocessing_context = 'spawn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)
    model = PredictLSTM(64, 1).to(device)
    # model.load_state_dict(torch.load('submission/encoder_dilate.pt'))
    optimiser = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    criterion = MS_SSIMLoss(channels=N_IMS)
    losses = []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        running_loss = 0
        i = 0
        count = 0
        for batch_coordinates, batch_features, batch_targets in ch_dataloader:
            # for j in range(12 - 3):
            #print('MAX VALUE', torch.max(batch_features))
            optimiser.zero_grad()
            batch_features = rearrange(batch_features, 'b (t c) h w -> b t c h w', c=1) 
            feature = batch_features.to(device)
            batch_predictions = model(feature, future_seq=24)#[::, ::, 32:96, 32:96]#.to(device)
            batch_predictions = rearrange(batch_predictions, 'b t c h w -> b (t c) h w')
            # print(batch_targets.shape)
            batch_loss = criterion(batch_predictions.unsqueeze(dim=2), batch_targets.to(device).unsqueeze(dim=2))
            batch_loss.backward()
            optimiser.step()
            running_loss += batch_loss.item() * batch_predictions.shape[0]
            count += batch_predictions.shape[0]
            i += 1
            print(f"Completed batch {i} of epoch {epoch + 1} with loss {batch_loss.item()} -- processed {count} image sequences ({12 * count} images)")
        losses.append(running_loss / count)
        print(f"Loss for epoch {epoch + 1}/{EPOCHS}: {losses[-1]}")

    torch.save(model.state_dict(), f'submission/lstm_nolight.pt')
