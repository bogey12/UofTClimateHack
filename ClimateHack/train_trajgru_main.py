

import numpy as np
import random
import copy
import gc
import torch
import torch.optim as optim
import torch.nn as nn
import random
import os
import argparse
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_msssim import ms_ssim, MS_SSIM
from torchsummary import summary
from training_config import add_arguments
from dataset import ClimateHackDataset
from train_trajgru import TempModel, MS_SSIMLoss
from processing_utils import *


BATCH_SIZE = 64
EPOCHS = 1000
#MEAN = 0.3028213
#STDDEV = 0.16613534
in_channels = 12
out_channels = 24
lag = 0
random.seed(182739)

val_features = np.load("features.npz")
val_targets = np.load("targets.npz")
tensor_x = np.array(val_features["data"])
tensor_y = np.array(val_targets["data"])
tensor_x = torch.Tensor(tensor_x)
tensor_y = torch.Tensor(tensor_y)
test_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

def test(model, device, config):
    scores = []
    test_criterion = MS_SSIM(data_range=1023.0, size_average=True, win_size=3, channel=1)
    model.eval()
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            y, extra = preprocessing(config, batch_features)
            batch_predictions = model(y) # out: (b,24,128,128) UNET3d: (b,2,12,128,128)
            batch_predictions = postprocessing(config, batch_predictions, extra)
            batch_loss = test_criterion(batch_predictions.unsqueeze(2).squeeze(0), batch_targets.unsqueeze(2).squeeze(0))
            scores.append(batch_loss.cpu().numpy())
    print(f"Test Score: {np.mean(scores)} ({np.std(scores)})")
    model.train()
    return np.mean(scores)

def train(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TempModel(config)
    model.load_state_dict(torch.load("submission/Test_TrajGRU"))
    model = model.to(device)
    #print(summary(model, (12, 128, 128)))
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    #optimiser = optim.AdamW(model.parameters(), lr=5e-4)
    MSSIM_criterion = MS_SSIMLoss(channels=out_channels, data_range=1023)

    test_score = test(model,device,config)
    best_test_score = test_score

    losses = []
    val_losses = []
    best_val_score = 0
    best_epoch = -1
    best_model = None
    data_dir = "/prj/qct/yyz-ml-users/data/Satellite"
    file_list = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    random.shuffle(file_list)
    #train_files = file_list[0:round(0.85*len(file_list))]
    train_files = file_list[0:len(file_list)]
    val_files = [file_list[len(file_list)-1]]
    #val_files = file_list[round(0.85*len(file_list)):len(file_list)]
    for epoch in range(EPOCHS):
        running_loss = 0
        val_loss = 0
        i = 0
        count = 0
        print("Epoch {}".format(epoch + 1))
        random.shuffle(train_files)
        random.shuffle(val_files)
        for file in train_files:
            ch_dataset = ClimateHackDataset(os.path.join(data_dir,file), crops_per_slice=10, in_channels=in_channels, out_channels=out_channels, lag=lag)
            py_dataset = ch_dataset.get_trajgru_dataset()
            dataloader = DataLoader(py_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
            for batch_features, batch_targets in dataloader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                optimiser.zero_grad()
                y, extra = preprocessing(config, batch_features)
                batch_predictions = model(y) # out: (b,24,128,128) UNET3d: (b,2,12,128,128)
                batch_predictions = postprocessing(config, batch_predictions, extra)
                batch_loss = MSSIM_criterion(batch_predictions.unsqueeze(dim=2), batch_targets[:,:24].unsqueeze(dim=2))
                batch_loss.backward()
                optimiser.step()
                running_loss += batch_loss.item() * batch_predictions.shape[0]
                count += batch_predictions.shape[0]
                if ((i+1) % 5000 == 0):
                    print("Batch {} running loss {}".format(i+1, running_loss/count))
                i += 1
            del ch_dataset,py_dataset,dataloader
        #losses.append(running_loss / count)
        #print("Loss for epoch {}: {}".format(epoch + 1, running_loss/count))
        test_score = test(model,device,config)
        if test_score > best_test_score:
            best_test_score = test_score
            torch.save(model.state_dict(), "submission/Test_TrajGRU_BS64lr1e4.pth")
            print("saving best test model")
        gc.collect()
    print("Best Model: Epoch = {}, Val MS_SSIM Score = {}".format(best_epoch,best_val_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train skip conn UNet')
    add_arguments(parser)
    args = vars(parser.parse_args())
    print(args)
    train(args)