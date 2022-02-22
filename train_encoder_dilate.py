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
# plt.rcParams["figure.figsize"] = (20, 12)
BATCH_SIZE = 1
EPOCHS = 10
SATELLITE_ZARR_PATH = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
N_IMS = 24
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
    ch_dataset = ClimateHackDataset(dataset, crops_per_slice=5, day_limit=7, outputs=N_IMS, timeskip=10)
    datapoints = pickle.load(open('datapoints.pkl', 'rb'))
    ch_dataset.cached_items = datapoints

    ch_dataloader = DataLoader(ch_dataset,  batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    ch_dataloader.multiprocessing_context = 'spawn'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)
    model = Encoder(inputs=12, outputs=N_IMS, half_size=True).to(device)
    model.load_state_dict(torch.load('submission/encoder_dilate.pt'))

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
            feature = batch_features.to(device)
            batch_predictions = model(feature)#[::, ::, 32:96, 32:96]#.to(device)
            # print(batch_targets.shape)
            batch_loss = criterion(batch_predictions.unsqueeze(dim=2), batch_targets.to(device).unsqueeze(dim=2))
            batch_loss.backward()
            optimiser.step()
            running_loss += batch_loss.item() * batch_predictions.shape[0]
            count += batch_predictions.shape[0]
            i += 1
            # print(f"Completed batch {i} of epoch {epoch + 1} with loss {batch_loss.item()} -- processed {count} image sequences ({12 * count} images)")
        losses.append(running_loss / count)
        print(f"Loss for epoch {epoch + 1}/{EPOCHS}: {losses[-1]}")


    
    # scorer = MS_SSIM(win_size=3, channel=3)
    # for datapoint in ch_dataset:
    #     coords, input_data, output_data = datapoint
    #     for j in range(12 - 3):
    #         fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(20, 8))
    #         im_sequence = torch.from_numpy(input_data[j:j+3, ::, ::]).view(1, 3, 128, 128).to(device)
    #         predicted_tensor = model(im_sequence).detach().to(device)
    #         predicted = predicted_tensor.numpy().reshape(3, 128, 128)
    #         print('SCORE:', scorer(predicted_tensor, im_sequence))
    #         for i, im in enumerate(input_data[j:j+3, ::, ::]):
    #             ax1[i].imshow(im, cmap='viridis')
    #             ax1[i].get_xaxis().set_visible(False)
    #             ax1[i].get_yaxis().set_visible(False)

    #         for i, im in enumerate(predicted):
    #             ax2[i].imshow(im, cmap='viridis')
    #             ax2[i].get_xaxis().set_visible(False)
    #             ax2[i].get_yaxis().set_visible(False)
            
    #         fig.show()
    #         input()
    
    torch.save(model.state_dict(), f'submission/encoder_dilate.pt')
