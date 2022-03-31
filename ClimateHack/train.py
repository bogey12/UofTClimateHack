#import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow import keras
#import matplotlib.pyplot as plt
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
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from numpy import float32
from dataset import ClimateHackDataset
from loss import MS_SSIMLoss 
from submission import ConvLSTMModel, UNET, UNET64, UNET_ConvGRU, UNETViT, ViT_Net, Focal_UNet, Focal_FullUNet
#from metnet import MetNet
from pytorch_msssim import ms_ssim, MS_SSIM
from torchsummary import summary
from TransUNet_Models import AA_TransUnet


# CUDA art path: /local/mnt2/workspace2/tol/anaconda3/envs/climatehack/lib
#plt.rcParams["figure.figsize"] = (20, 12)
BATCH_SIZE = 64
EPOCHS = 1000
MEAN = 0.3028213
STDDEV = 0.16613534
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
            # Permute batch_predictions: (b,1,12,64,64) -> (b,12,1,64,64)
            #batch_features = batch_features.permute(0, 2, 1, 3, 4)
            # Normalize batch_predictions before feeding to model again
            #new_features = (batch_predictions - MEAN) / STDDEV
            #batch_features = torch.cat((batch_features[:,1:12,:,:,:], new_features), axis=1)
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
        #print(f"Validation Loss: {validation_loss}")
        #print(f"MS_SSIM SCORE: {validation_loss}")
    model.train()
    return validation_loss

def test_seq2one(model,criterion,device):
    dataset = ClimateHackDataset("../data2.npz", crops_per_slice=1, in_channels=12, out_channels=24)
    a_dataset = dataset.get_dataset()
    #_, val_ds = random_split(a_dataset, [len(a_dataset)-500, 500])
    dataloader = DataLoader(a_dataset, batch_size=1, shuffle=False)
    #model = model.to(device)
    single_criterion = MS_SSIM(data_range=1.0, size_average=False, win_size=3, channel=1)
    running_loss = 0
    i = 0
    count = 0
    with torch.no_grad():
        for batch_features, batch_targets in dataloader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            #batch_features = batch_features[:,:,:,32:96,32:96]
            # shape (1,1,12,64,64)
            # MetNet Train Flow (permute from (1,1,12,128,128) -> (1,12,1,128,128))
            batch_features = batch_features.permute(0, 2, 1, 3, 4)
            batch_predictions = []
            cur_in = batch_features
            for i in range(24):
                temp_out = model(cur_in, 0)
                # predictions shape (1,1,64,8,8) -> (1,1,1,64,64)
                temp_out = temp_out.view(-1,1,1,64,64)
                print(single_criterion(temp_out.squeeze(1), batch_targets[:,:,i,:,:]).item())
                batch_predictions.append(temp_out)
                # resize temp out to (1,1,1,128,128)
                #cur_out = expand(temp_out.squeeze(0)).unsqueeze(0)
                # concatenate (1,11,1,128,128) with (1,1,1,128,128) on axis 1
                cur_in = torch.cat((cur_in[:,1:12,:,:,:], temp_out), axis=1)
            batch_predictions = torch.stack(batch_predictions, dim=1)
            batch_predictions = 1023*batch_predictions.squeeze().unsqueeze(0)
            batch_targets = batch_targets.squeeze().unsqueeze(0)
            #print(batch_targets.shape)
            #print(batch_predictions.shape)
            print("BREAK")
            batch_loss = criterion(batch_predictions,batch_targets)
            running_loss += batch_loss.item() * batch_predictions.shape[0]
            count += batch_predictions.shape[0]
            i += 1
    validation_loss = running_loss / count
    #print(f"Validation Loss: {validation_loss}")
    #print(f"MS_SSIM SCORE: {validation_loss}")
    return validation_loss

def train():
    hparams={'img_dim': 128, 'in_channels': 12, 'out_channels': 24, 'vit_blocks': 1, 'vit_heads': 1, 'vit_dim_linear_mhsa_block': 3072, 'vit_transformer_dim': 1024, 'patch_size': 2, 'vit_channels': None, 'vit_transformer': None}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #model = ConvLSTMModel(time_steps=1,channels=in_channels,hidden_dim=64,out_steps=out_channels)
    '''
    model = MetNet(
            hidden_dim=64,
            forecast_steps=out_channels,
            input_channels=1,
            output_channels=64, # Change for Out Size
            sat_channels=1,
            input_size=32, # change for seq2seq
            )
    '''
    #model = UNET(in_channels=12, out_channels=24) #
    #model = torch.nn.DataParallel(UNET_ConvGRU(12,24), device_ids=[0, 1, 2, 3])
    model = UNETViT(12,24)
    #model = Focal_UNet()
    #model = Focal_FullUNet()
    #model = AA_TransUnet(hparams)
    #model = UNET64(umodel)
    model.load_state_dict(torch.load("submission/Test_UNet_ViT_FullData.pth"))
    model = model.to(device)

    # Switch the model to eval model
    #model.eval()

    # An example input you would normally provide to your model's forward() method.
    #example = torch.rand(2, 12, 128, 128).to(device)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    #traced_script_module = torch.jit.trace(model, example)

    # Save the TorchScript model
    #traced_script_module.save("UNet_ViT_CBAM_proj12_ResizeConv.pt")
    #print(summary(model, (12, 128, 128)))
    print(sum(p.numel() for p in model.parameters()))

    #torch.save(model.state_dict(), "submission/TransUNet.pth")
    #print(x)
    #optimiser = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimiser = optim.Adam(model.parameters(), lr=2.8e-4)
    criterion = nn.MSELoss()
    MSSIM_criterion = MS_SSIM(data_range=1.0, size_average=True, win_size=3, channel=out_channels)


    #MSSIM_loss = MS_SSIMLoss(channels=out_channels)
    #test_seq2one(model,MSSIM_criterion,device)
    test_score = test(model,device)
    best_test_score = test_score
    #torch.save(model.state_dict(), "submission/Test_FocalNet64_embed48_0Drop.pth")
    print("saving best test model")

    losses = []
    val_losses = []
    best_val_score = 0
    best_epoch = -1
    best_model = None
    #val_loss = eval(model, MSSIM_loss, dataloader, device)
    data_dir = "/prj/qct/yyz-ml-users/data/Satellite"
    file_list = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    random.shuffle(file_list)
    #train_files = file_list[0:round(0.85*len(file_list))]
    train_files = file_list[0:len(file_list)]
    val_files = [file_list[len(file_list)-1]]
    #val_files = file_list[round(0.85*len(file_list)):len(file_list)]

    transformations = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)])

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
            #train_len = round(0.8*len(py_dataset))
            #val_len = len(py_dataset) - train_len
            #train_ds, val_ds = random_split(py_dataset, [train_len, val_len]) # 0.8 train, 0.2 val
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
                #batch_loss = 1 - MSSIM_criterion(batch_predictions, batch_targets)
                batch_loss.backward()
                optimiser.step()
                running_loss += batch_loss.item() * batch_predictions.shape[0]
                count += batch_predictions.shape[0]
                #print(running_loss/count)
                if ((i+1) % 5000 == 0):
                    print("Batch {} running loss {}".format(i+1, running_loss/count))
                i += 1
            #val_loss += eval(model, MSSIM_criterion, valloader, device)
            del ch_dataset,py_dataset,dataloader
        #for valfile in val_files:
        #    ch_dataset = ClimateHackDataset(os.path.join(data_dir,valfile), crops_per_slice=10, in_channels=in_channels, out_channels=out_channels, lag=lag)
        #    py_dataset = ch_dataset.get_dataset()
        #    valloader = DataLoader(py_dataset, batch_size=BATCH_SIZE, shuffle=False)
        #    val_loss += eval(model, MSSIM_criterion, valloader, device)
        #    del ch_dataset,py_dataset,valloader
        losses.append(running_loss / count)
        print("Loss for epoch {}: {}".format(epoch + 1, running_loss/count))
        #print("Validation Score for Epoch {}: {}".format(epoch + 1, val_loss / len(val_files)))
        #val_loss = 1 - (running_loss / count)
        #val_losses.append(val_loss / len(val_files))
        #if val_loss > best_val_score:
        #    best_val_score = val_loss
        #    best_epoch = epoch
            #best_model = copy.deepcopy(model)
            #torch.save(best_model.state_dict(), "submission/UNet_Resize_cproj12groups_FullData.pth")
            #print("saving best model at epoch x")
        test_score = test(model,device)
        if test_score > best_test_score:
            best_test_score = test_score
            torch.save(model.state_dict(), "submission/Test_UNet_ViT_V2.pth")
            print("saving best test model")
        gc.collect()
    print("Best Model: Epoch = {}, Val MS_SSIM Score = {}".format(best_epoch,best_val_loss))

if __name__ == '__main__':
    train()