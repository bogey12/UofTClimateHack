import numpy as np
import random
import copy
import gc
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import random
import os
from torch.utils.data import DataLoader, random_split, TensorDataset
from numpy import float32
from dataset import ClimateHackDataset
from loss import MS_SSIMLoss 
from submission import ConvLSTMModel, UNET, UNET64, UNET_ConvGRU, UNETViT, ViT_Net, Focal_UNet, Focal_FullUNet
from pytorch_msssim import ms_ssim, MS_SSIM
from torchsummary import summary

# Training hyperparams and constants
BATCH_SIZE = 64
EPOCHS = 1000
MEAN = 0.3028213
STDDEV = 0.16613534
in_channels = 12
out_channels = 24
lag = 0
random.seed(182739)

# Load local validator samples as test data. Found that local validator samples were good approximator of test set performance.
val_features = np.load("features.npz")
val_targets = np.load("targets.npz")
tensor_x = np.array(val_features["data"])
tensor_y = np.array(val_targets["data"])
tensor_x = torch.Tensor(tensor_x)
tensor_y = torch.Tensor(tensor_y)
test_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

def test(model, device):
    scores = []
    test_criterion = MS_SSIM(data_range=1023.0, size_average=True, win_size=3, channel=1)
    model.eval()
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features / 1023
            batch_features = (batch_features - MEAN) / STDDEV
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            batch_predictions = model(batch_features)
            batch_predictions = 1023 * batch_predictions[:,:,32:96,32:96]
            #batch_predictions = 1023 * batch_predictions
            batch_loss = test_criterion(batch_predictions.unsqueeze(2).squeeze(0), batch_targets.unsqueeze(2).squeeze(0))
            scores.append(batch_loss.cpu().numpy())
    print(f"Test Score: {np.mean(scores)} ({np.std(scores)})")
    model.train()
    return np.mean(scores)

def eval(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        count = 0
        for batch_features, batch_targets in dataloader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            #batch_features = batch_features[:,:,32:96,32:96]
            # Repermute batch_predictions: (b,24,1,64,64) -> (b,1,24,64,64)
            batch_features = batch_features.squeeze(2)
            batch_predictions = model(batch_features) # out: (b,24,128,128)
            #batch_predictions = batch_predictions.squeeze(2)
            #batch_predictions = batch_predictions.view(-1,1,24,128,128).squeeze(1)
            batch_predictions = batch_predictions[:,:,32:96,32:96]
            batch_loss = criterion(batch_predictions, batch_targets)
            #batch_loss = MSSIM_criterion(batch_predictions, batch_targets)
            running_loss += batch_loss.item() * batch_predictions.shape[0]
            count += batch_predictions.shape[0]
        validation_loss = running_loss / count
    model.train()
    return validation_loss

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #model = UNET(in_channels=12, out_channels=24) #
    model = UNETViT(12,24)
    #model = Focal_UNet()
    model.load_state_dict(torch.load("submission/Test_UNet_ViT_FullData.pth"))
    model = model.to(device)

    #print(summary(model, (12, 128, 128)))
    print(sum(p.numel() for p in model.parameters()))

    optimiser = optim.Adam(model.parameters(), lr=2.8e-4)
    MSSIM_criterion = MS_SSIM(data_range=1.0, size_average=True, win_size=3, channel=out_channels)

    test_score = test(model,device)
    best_test_score = test_score
    #torch.save(model.state_dict(), "submission/Test_FocalNet64_embed48_0Drop.pth")
    print("saving best test model")

    losses = []
    val_losses = []
    best_val_score = 0
    best_epoch = -1
    best_model = None
    data_dir = "/prj/qct/yyz-ml-users/data/Satellite"
    file_list = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    random.shuffle(file_list)

    # 0.85/0.15 train val split
    train_files = file_list[0:round(0.85*len(file_list))]
    #train_files = file_list[0:len(file_list)]
    #val_files = [file_list[len(file_list)-1]]
    val_files = file_list[round(0.85*len(file_list)):len(file_list)]
    # Recrop from satellite images for each epoch to prevent overfitting
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
            py_dataset = ch_dataset.get_dataset()
            dataloader = DataLoader(py_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
            for batch_features, batch_targets in dataloader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                optimiser.zero_grad()
                # UNET: features -> (b,12,128,128)
                # UNET_ConvGRU: features -> (b,12,1,128,128)
                batch_features = batch_features.squeeze(2)
                batch_predictions = model(batch_features) # out: (b,24,128,128) UNET3d: (b,2,12,128,128)
                #batch_predictions = batch_predictions.squeeze(2)
                #batch_predictions = batch_predictions.view(-1,1,24,128,128).squeeze(1)
                batch_predictions = batch_predictions[:,:,32:96,32:96]
                batch_loss = 1 - MSSIM_criterion(batch_predictions, batch_targets)
                batch_loss.backward()
                optimiser.step()
                running_loss += batch_loss.item() * batch_predictions.shape[0]
                count += batch_predictions.shape[0]
                #print(running_loss/count)
                if ((i+1) % 5000 == 0):
                    print("Batch {} running loss {}".format(i+1, running_loss/count))
                i += 1
            del ch_dataset,py_dataset,dataloader
        for valfile in val_files:
            ch_dataset = ClimateHackDataset(os.path.join(data_dir,valfile), crops_per_slice=10, in_channels=in_channels, out_channels=out_channels, lag=lag)
            py_dataset = ch_dataset.get_dataset()
            valloader = DataLoader(py_dataset, batch_size=BATCH_SIZE, shuffle=False)
            val_loss += eval(model, MSSIM_criterion, valloader, device)
            del ch_dataset,py_dataset,valloader
        losses.append(running_loss / count)
        print("Loss for epoch {}: {}".format(epoch + 1, running_loss/count))
        print("Validation Score for Epoch {}: {}".format(epoch + 1, val_loss / len(val_files)))
        #val_loss = 1 - (running_loss / count)
        #val_losses.append(val_loss / len(val_files))
        #if val_loss > best_val_score:
        #    best_val_score = val_loss
        #    best_epoch = epoch
            #best_model = copy.deepcopy(model)
            #torch.save(best_model.state_dict(), "submission/UNet_Resize_cproj12groups_FullData.pth")
            #print("saving best model at epoch x")
        # save best test model
        test_score = test(model,device)
        if test_score > best_test_score:
            best_test_score = test_score
            torch.save(model.state_dict(), "submission/Test_UNet_ViT_V2.pth")
            print("saving best test model")
        gc.collect()
    print("Best Model: Epoch = {}, Val MS_SSIM Score = {}".format(best_epoch,best_val_loss))

if __name__ == '__main__':
    train()