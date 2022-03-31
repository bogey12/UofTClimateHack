import os
from datetime import datetime, time, timedelta
from functools import lru_cache
from random import randrange, seed
from typing import Iterator, T_co
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import numpy as np
#import xarray as xr
from numpy import float32

class ClimateHackDataset():
    def __init__(
        self,
        data_path: str,
        crops_per_slice: int = 1,
        in_channels: int = 12,
        out_channels: int = 24,
        lag: int = 0
    ) -> None:
        self.crops_per_slice = crops_per_slice
        self.coordinates = []
        self.features = []
        self.labels = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lag = lag
        #seed(1557)
        self._process_data(np.load(data_path), self.in_channels, self.out_channels)
        
    def load_data(self):
        return np.array(self.coordinates), np.array(self.features), np.array(self.labels)

    def save_data(self, name):
        mean = 0.3028213
        stddev = 0.16613534
        tensor_x = np.array(self.features)
        print(tensor_x.shape)
        tensor_y = np.array(self.labels)
        tensor_x = tensor_x / 1023
        tensor_x = (tensor_x - mean) / stddev
        tensor_y = tensor_y / 1023
        np.savez(name, tensor_x, tensor_y, x=tensor_x, y=tensor_y)
        return True   

    def get_dataset(self):
        mean = 0.3028213
        stddev = 0.16613534
        tensor_x = np.array(self.features)
        tensor_y = np.array(self.labels)
        tensor_x = torch.Tensor(tensor_x).unsqueeze(2) # Channels=12, time_steps=1
        tensor_y = torch.Tensor(tensor_y) #
        tensor_x = tensor_x / 1023
        tensor_x = (tensor_x - mean) / stddev
        tensor_y = tensor_y / 1023
        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        return my_dataset

    def get_trajgru_dataset(self):
        tensor_x = np.array(self.features)
        tensor_y = np.array(self.labels)
        tensor_x = torch.Tensor(tensor_x) # Channels=12, time_steps=1
        tensor_y = torch.Tensor(tensor_y) #
        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        return my_dataset
    def _process_data(self, data, in_channels, out_channels):
        day = data
        # day = (day - mean) / stddev
        for i in range(0, day.shape[0] - (in_channels+out_channels+self.lag) + 1, 1):
            input_slice = day[i : i + in_channels, :, :]
            target_slice = day[i + in_channels +self.lag: i + (in_channels + out_channels)+self.lag, :, :]
            crops = 0
            while crops < self.crops_per_slice:
                crop = self._get_crop(input_slice, target_slice)
                if crop:
                    (input_data, target_data) = crop
                    self.features += input_data,
                    self.labels += target_data,

                crops += 1

    def _get_crop(self, input_slice, target_slice):
        # roughly over the mainland UK
        rand_x = randrange(550, 950 - 128)
        rand_y = randrange(375, 700 - 128)

        # get the input satellite imagery
        input_data = input_slice[:, rand_y : rand_y + 128, rand_x : rand_x + 128]
        target_data = target_slice[
            :, rand_y + 32 : rand_y + 96, rand_x + 32 : rand_x + 96 #(size 64: 32 - 96)
        ]

        if input_data.shape != (self.in_channels, 128, 128) or target_data.shape != (self.out_channels, 64, 64):
            return None

        return input_data, target_data

class Climate_dataset(Dataset):
    def __init__(self, data_dir):
        # Compute chunk sizes
        self.num_exs_per_file = 190
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir)]
        self.cur_file = 0
        self.MEAN = 0.3028213
        self.STDDEV = 0.16613534
        self.get_data(self.files[0])
    
    def get_data(self, filename):
        #npzfile = np.load(os.path.join(self.data_dir,filename))
        #tensor_x = npzfile['x']
        #tensor_y = npzfile['y']
        chDataset = ClimateHackDataset(os.path.join(self.data_dir,filename), crops_per_slice=10, in_channels=12, out_channels=24, lag=0)
        tensor_x = np.array(chDataset.features)
        tensor_y = np.array(chDataset.labels)
        tensor_x = torch.Tensor(tensor_x) # Channels=12, time_steps=1
        tensor_y = torch.Tensor(tensor_y) #
        tensor_x = tensor_x / 1023
        self.x_data = (tensor_x - self.MEAN) / self.STDDEV
        self.y_data = tensor_y / 1023

    def __len__(self): 
        return self.num_exs_per_file * len(self.files)
    
    def __getitem__(self, idx):
        file_idx = idx // self.num_exs_per_file
        exs_idx = idx % self.num_exs_per_file
        if self.cur_file != file_idx:
            self.get_data(self.files[file_idx])
            self.cur_file = file_idx
        return self.x_data[exs_idx], self.y_data[exs_idx]

if __name__ == '__main__':
    data_dir = "/prj/qct/yyz-ml-users/data/Satellite"
    out_dir = "/prj/qct/yyz-ml-users/data/SatelliteV2"
    for f in os.listdir(out_dir):
        npzfile = np.load(os.path.join(out_dir,f))
        size = npzfile['x'].shape[0]
        if size != 190:
            print(f)
            print(size)

    #train_files = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    #print(len(train_files))
    #for i in range(0,len(train_files)):
    #    print(train_files[i])
    #    ch_dataset = ClimateHackDataset(os.path.join(data_dir,train_files[i]), crops_per_slice=5, in_channels=12, out_channels=24, lag=0)
    #    ch_dataset.save_data(os.path.join(out_dir,"file_{}".format(i)))