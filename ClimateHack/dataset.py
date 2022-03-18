from datetime import datetime, time, timedelta
from functools import lru_cache
from random import randrange, seed
from typing import Iterator, T_co
from torch.utils.data import TensorDataset, DataLoader
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

    def get_dataset(self):
        tensor_x = np.array(self.features)
        tensor_y = np.array(self.labels)
        tensor_x = torch.Tensor(tensor_x).unsqueeze(2) # Channels=12, time_steps=1
        tensor_y = torch.Tensor(tensor_y) # 
        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        return my_dataset

    def _process_data(self, data, in_channels, out_channels):
        #print("Processing...")
        '''
        self.osgb_data = np.stack(
            [
                data["x_osgb"],
                data["y_osgb"],
            ]
        )
        
        #alldata = None
       
        flag = False
        for day in data["data"]:
            if flag == False:
                alldata = day
                flag = True
            else:
                alldata = np.concatenate((alldata,day))
        #print(alldata.shape)
        # Scale data from [0,1023] to [0,1]
        alldata = alldata / 1023
        # Get mean, stddev across all channels and standardize data
        mean = np.mean(alldata)
        stddev = np.std(alldata)
        print(mean)
        print(stddev)
        '''
        #data["data"] = (data["data"] - mean) / stddev
        mean = 0.3028213
        stddev = 0.16613534
        #print(data.shape)
        #print(data["data"])
        day = data / 1023
        # day = (day - mean) / stddev
        for i in range(0, day.shape[0] - (in_channels+out_channels+self.lag) + 1, 1):
            input_slice = day[i : i + in_channels, :, :]
            target_slice = day[i + in_channels +self.lag: i + (in_channels + out_channels)+self.lag, :, :]
            crops = 0
            while crops < self.crops_per_slice:
                crop = self._get_crop(input_slice, target_slice)
                if crop:
                    (input_data, target_data) = crop
                    temp_data = (input_data - mean) / stddev
                    self.features += temp_data,
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

#     def __iter__(self) -> Iterator[T_co]:
#         for item in self.cached_items:
#             yield item
