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
from pytorch_msssim import MS_SSIM
# from submission.model import Model
import numpy as np
import cv2
from submission.model import *
# plt.rcParams["figure.figsize"] = (20, 12)
BATCH_SIZE = 1
EPOCHS = 1
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
N_IMS = 3
# print(dataset)

def average_flows(input_data, n_warmup, **args):
    flows = []
    for i in range(n_warmup):
        flows.append(cv2.calcOpticalFlowFarneback(prev=input_data[i], next=input_data[i+1], flow=None, **args))
    flows = np.stack(flows).astype(float32)
    return np.average(flows, axis=0, weights=range(1, n_warmup+1)).astype(np.float32)

def remap_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    # Adapted from https://github.com/opencv/opencv/issues/11068
    height, width = flow.shape[:2]
    remap = -flow.copy()
    remap[..., 0] += np.arange(width)  # x map
    remap[..., 1] += np.arange(height)[:, np.newaxis]  # y map
    return cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

if __name__ == '__main__':
    # freeze_support()
    dataset = xr.open_dataset(
        SATELLITE_ZARR_PATH, 
        engine="zarr",
        chunks="auto",  # Load the data as a Dask array
    )
    ch_dataset = ClimateHackDataset(dataset, crops_per_slice=1, day_limit=7, outputs=3)
    ch_dataloader = DataLoader(ch_dataset, batch_size=BATCH_SIZE, num_workers=0)
    # ch_dataloader.multiprocessing_context = 'spawn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)
    # model = Encoder(inputs=N_IMS, outputs=N_IMS).to(device)
    # model = torch.load('submission/encoder2.pt')
    scorer = MS_SSIM(win_size=3, channel=3)
    # optimiser = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    # criterion = MS_SSIMLoss(channels=3)
    with torch.no_grad():
        for datapoint in ch_dataset:
            coords, input_data, output_data = datapoint
            # for j in range(12 - 3):
            fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(20, 8))
            # print(input_data[-3:, ::, ::].shape)
            print(output_data.shape)
            im_sequence = torch.tensor(input_data[-3:, ::, ::]).view(1, 3, 128, 128)#.to(device)
            predicted_tensor = model(im_sequence).detach()#.to(device)
            print(predicted_tensor.shape)
            # print(im_sequence)
            predicted = predicted_tensor[::, ::, 32:96, 32:96].numpy().reshape(3, 64, 64)
            print('SCORE:', scorer(predicted_tensor[::, ::, 32:96, 32:96], torch.tensor(output_data).view(1, 3, 64, 64)))
            for i, im in enumerate(output_data[:3, ::, ::]):
                ax1[i].imshow(im, cmap='viridis')
                ax1[i].get_xaxis().set_visible(False)
                ax1[i].get_yaxis().set_visible(False)

            for i, im in enumerate(predicted):
                ax2[i].imshow(im, cmap='viridis')
                ax2[i].get_xaxis().set_visible(False)
                ax2[i].get_yaxis().set_visible(False)
            
            fig.show()
            input()
