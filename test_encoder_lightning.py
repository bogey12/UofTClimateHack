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

STATE_PATH = 'submission/encoder_generic.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def_model = Encoder(inputs=12, outputs=24, fac=1023, half_size=True, dropout=0).to(device)
# model = PredictionTrainer(model=def_model, device=device, convert=False)
model = ModelWrapper(model=def_model)
model.load_state_dict(torch.load(STATE_PATH))
model.eval()

# encoder = [('conv', 'leaky', 1, 32, 3, 1, 2),
#             ('convlstm', '', 32, 32, 3, 1, 1),
#             ('conv', 'leaky', 32, 64, 3, 1, 2),
#             ('convlstm', '', 64, 64, 3, 1, 1),
#             ('conv', 'leaky', 64, 128, 3, 1, 2),
#             ('convlstm', '', 128, 128, 3, 1, 1)]
# decoder = [('deconv', 'leaky', 128, 64, 4, 1, 2),
#             ('convlstm', '', 128, 64, 3, 1, 1),
#             ('deconv', 'leaky', 64, 32, 4, 1, 2),
#             ('convlstm', '', 64, 32, 3, 1, 1),
#             ('deconv', 'leaky', 32, 32, 4, 1, 2),
#             ('convlstm', '', 33, 32, 3, 1, 1),
#             ('conv_output', '', 32, 1, 1, 0, 2)]
# encoder = [('conv', 'leaky', 1, 32, 5, 2, 2),
#              ('convlstm', '', 32, 32, 3, 1, 1),
#              ('conv', 'leaky', 32, 64, 3, 1, 2),
#              ('convlstm', '', 64, 64, 3, 1, 1),
#              ('conv', 'leaky', 64, 128, 3, 1, 2),
#              ('convlstm', '', 128, 128, 3, 1, 1)]
# decoder = [('deconv', 'leaky', 128, 64, 4, 1, 2),
#             ('convlstm', '', 128, 64, 3, 1, 1),
#             ('deconv', 'leaky', 64, 32, 4, 1, 2),
#             ('convlstm', '', 64, 32, 3, 1, 1),
#             ('deconv', 'leaky', 32, 32, 4, 1, 2),
#             ('convlstm', '', 33, 32, 3, 1, 1),
#             ('conv_output', '', 32, 1, 1, 0, 2)]
# STATE_PATH = 'submission/convlstm.pt'
# def_model = ConvLSTM(encoder, decoder)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = PredictionTrainer(model=def_model, device=device, convert=True, data_range=1023, channels=12, fac=1023)
# model.load_state_dict(torch.load(STATE_PATH))
# model.eval()

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
    test_ds = dataset.sel(time=slice("2020-10-01 09:00", "2020-11-01 09:00"))
    ch_ds = ClimateHackDataset(test_ds, crops_per_slice=5, day_limit=3, outputs=N_IMS)
    scorer = MS_SSIM(win_size=3, channel=24, data_range=1023)
    # scorer = MS_SSIM(win_size=3, channel=12, data_range=1023)
    with torch.no_grad():
        for datapoint in ch_ds:
            input_data, output_data = datapoint
            # for j in range(12 - 3):
            fig, (ax1, ax2) = plt.subplots(2, 12, figsize=(20, 8))
            # print(input_data[-3:, ::, ::].shape)
            # print(output_data.shape)
            im_sequence = torch.tensor(input_data).view(1, 12, 128, 128).to(device)
            # im_sequence = torch.tensor(input_data).view(1, 12,1, 128, 128).to(device)
            predicted_tensor = model(im_sequence).detach()#.to(device)
            print('PREDICTION', predicted_tensor)
            print('ACTUAL', torch.tensor(output_data))
            # print(im_sequence)
            predicted = predicted_tensor.cpu().numpy().reshape(24, 64, 64)
            # predicted = predicted_tensor.cpu().numpy().reshape(12, 64, 64)
            print('SCORE:', scorer(predicted_tensor.view(1, 24, 64, 64), torch.tensor(output_data).to(device).view(1, 24, 64, 64)))
            # print('SCORE:', scorer(predicted_tensor.view(1, 12, 64, 64), torch.tensor(output_data)[:12].to(device).view(1, 12, 64, 64)))
            for i, im in enumerate(output_data[:12, ::, ::]):
                ax1[i].imshow(im, cmap='viridis')
                ax1[i].get_xaxis().set_visible(False)
                ax1[i].get_yaxis().set_visible(False)

            for i, im in enumerate(predicted[:12, ::, ::]):
                ax2[i].imshow(im, cmap='viridis')
                ax2[i].get_xaxis().set_visible(False)
                ax2[i].get_yaxis().set_visible(False)
            
            fig.savefig('test.png')
            input()
