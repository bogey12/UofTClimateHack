from datetime import datetime, time, timedelta
from functools import lru_cache
import itertools
from random import randrange, randint, shuffle
from typing import Iterator, T_co

import numpy as np
import xarray as xr
from numpy import float32
from torch.utils.data import IterableDataset
import torch
import math

class ClimateHackDataset(IterableDataset):
    """
    This is a basic dataset class to help you get started with the Climate Hack.AI
    dataset. You are heavily encouraged to customise this to suit your needs.

    Notably, you will most likely want to modify this if you aim to train
    with the whole dataset, rather than just a small subset thereof.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        start_date: datetime = None,
        end_date: datetime = None,
        crops_per_slice: int = 0,
        day_limit: int = 0,
        outputs: int = 24,
        timeskip: int = 1,
        shuffler: bool = True,
        cache: bool = True
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.crops_per_slice = crops_per_slice
        self.day_limit = day_limit
        self.cached_items = []
        self.outputs = outputs
        times = self.dataset.get_index("time")
        self.min_date = times[0].date()
        self.max_date = times[-1].date()
        self.timeskip = timeskip
        self.shuffler = shuffler
        self.cache = cache
        if start_date is not None:
            self.min_date = max(self.min_date, start_date)

        if end_date is not None:
            self.max_date = min(self.max_date, end_date)
        elif self.day_limit > 0:
            self.max_date = min(
                self.max_date, self.min_date + timedelta(days=self.day_limit*self.timeskip)
            )

    def _image_times(self, start_time, end_time, min_date, max_date):
        date = min_date
        while date <= max_date:
            current_time = datetime.combine(date, start_time)
            while current_time.time() <= end_time:
                yield current_time
                current_time += timedelta(minutes=20)

            date += timedelta(days=self.timeskip)

    def _get_crop(self, input_slice, target_slice, x_range=None, y_range=None, grid_size=None, randomize=20):
        # roughly over the mainland UK
        x_range = x_range or (550, 950-128)
        y_range = y_range or (375, 700-128) 
        if grid_size is None:
            pairs = [(randrange(*x_range), randrange(*y_range))]
        else:
            x_range1 = list(range(*(x_range + (grid_size[0],))))
            shuffle(x_range1)
            y_range1 = list(range(*(y_range + (grid_size[1],))))
            shuffle(y_range1)
            pairs = itertools.product(x_range1, y_range1)
        
        # y_range = range(*y_range)
        for rand_x, rand_y in pairs:
            rand_x += randint(-min(rand_x - x_range[0], randomize), min(x_range[1] - rand_x, randomize))
            rand_y += randint(-min(rand_y - y_range[0], randomize), min(y_range[1] - rand_y, randomize))
            # print('PAIR:', rand_x, rand_y)
            # make a data selection
            selection = input_slice.isel(
                x=slice(rand_x, rand_x + 128),
                y=slice(rand_y, rand_y + 128),
            )

            # get the OSGB coordinate data
            osgb_data = np.stack(
                [
                    selection["x_osgb"].values.astype(float32),
                    selection["y_osgb"].values.astype(float32),
                ]
            )

            if osgb_data.shape != (2, 128, 128):
                continue

            # get the input satellite imagery
            input_data = selection["data"].values.astype(float32)
            if input_data.shape != (12, 128, 128):
                continue
            # get the target output
            target_output = (
                target_slice["data"]
                .isel(
                    x=slice(rand_x + 32, rand_x + 96),
                    y=slice(rand_y + 32, rand_y + 96),
                )
                .values.astype(float32)
            )

            if target_output.shape != (self.outputs, 64, 64):
                continue

            # yield osgb_data, input_data, target_output
            yield input_data, target_output

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()
        if self.cached_items:
            if worker_info:
                per_worker = math.ceil(len(self.cached_items)/worker_info.num_workers)
                cached_items = self.cached_items[per_worker*worker_info.id:per_worker*(worker_info.id + 1)]
            else:
                cached_items = self.cached_items
            if self.shuffler:
                shuffle(cached_items)
            for item in cached_items:
                yield item
            return
        else:
            start_time = time(9, 0)
            end_time = time(14, 0)
            max_date = self.max_date
            min_date = self.min_date
            if worker_info:
                day_len = (max_date - min_date).days
                per_worker = timedelta(days=math.ceil(day_len/worker_info.num_workers))
                min_date += worker_info.id*per_worker
                max_date = min_date + per_worker

            for current_time in self._image_times(start_time, end_time, min_date, max_date):
                #print(current_time)#, worker_info.id)
                data_slice = self.dataset.loc[
                    {
                        "time": slice(
                            current_time,
                            current_time + timedelta(hours=1) + timedelta(minutes=5)*(self.outputs - 1),
                        )
                    }
                ]

                if data_slice.sizes["time"] != 12 + self.outputs:
                    continue

                input_slice = data_slice.isel(time=slice(0, 12))
                target_slice = data_slice.isel(time=slice(12, 12 + self.outputs))
                if self.crops_per_slice != 0:
                    # print('CROP')
                    crops = 0
                    while crops < self.crops_per_slice:
                        for crop in self._get_crop(input_slice, target_slice):
                            if crop:
                                if self.cache:
                                    self.cached_items.append(crop)
                                yield crop
                        crops += 1
                else:
                    for crop in self._get_crop(input_slice, target_slice, grid_size=(64, 64)):
                        if self.cache:
                            self.cached_items.append(crop)
                        yield crop


class ClimateHackDataset2(IterableDataset):
    """
    This is a basic dataset class to help you get started with the Climate Hack.AI
    dataset. You are heavily encouraged to customise this to suit your needs.

    Notably, you will most likely want to modify this if you aim to train
    with the whole dataset, rather than just a small subset thereof.
    """

    def __init__(
        self,
        dataset,
        crops_per_slice: int = 0,
        outputs: int = 24,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.crops_per_slice = crops_per_slice
        self.cached_items = []
        self.outputs = outputs


    def _get_crop(self, input_slice, target_slice, x_range=None, y_range=None, grid_size=None, randomize=20):
        # roughly over the mainland UK
        x_range = x_range or (550, 950-128)
        y_range = y_range or (375, 700-128) 
        if grid_size is None:
            pairs = [(randrange(*x_range), randrange(*y_range))]
        else:
            x_range1 = list(range(*(x_range + (grid_size[0],))))
            shuffle(x_range1)
            y_range1 = list(range(*(y_range + (grid_size[1],))))
            shuffle(y_range1)
            pairs = itertools.product(x_range1, y_range1)
        
        # y_range = range(*y_range)
        for rand_x, rand_y in pairs:
            rand_x += randint(-min(rand_x - x_range[0], randomize), min(x_range[1] - rand_x, randomize))
            rand_y += randint(-min(rand_y - y_range[0], randomize), min(y_range[1] - rand_y, randomize))
            # print('PAIR:', rand_x, rand_y)
            # make a data selection
            input_data = input_slice[:, rand_x:rand_x + 128, rand_y:rand_y + 128]
            # get the input satellite imagery
            if input_data.shape != (12, 128, 128):
                continue
            # get the target output
            target_output = target_slice[:, rand_x+32:rand_x+96, rand_y+32:rand_y+96].astype(float32)
            if target_output.shape != (self.outputs, 64, 64):
                continue
            yield input_data, target_output

    def __iter__(self) -> Iterator[T_co]:
        for day in range(self.dataset.shape[0]):
            #print(current_time)#, worker_info.id)
            for time_slice in range(self.dataset.shape[1] - 12 - self.outputs, 4):
                input_slice = self.dataset[day][time_slice:time_slice+12]
                target_slice = self.dataset[day][time_slice+12:time_slice+12+self.outputs]
                if self.crops_per_slice != 0:
                    crops = 0
                    while crops < self.crops_per_slice:
                        for crop in self._get_crop(input_slice, target_slice):
                            if crop:
                                self.cached_items.append(crop)
                                yield crop
                        crops += 1
                else:
                    for crop in self._get_crop(input_slice, target_slice, grid_size=(128, 128)):
                        self.cached_items.append(crop)
                        yield crop