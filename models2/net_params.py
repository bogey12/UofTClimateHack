import sys
# sys.path.insert(1, 'C:\\Users\\HECAI\\Documents\\Personal\\ClimateAI\\grid_climate\\models2')
sys.path.insert(1, './models2')
from config import cfg
import torch
from forecaster import Forecaster
from encoder import Encoder
from collections import OrderedDict
from model import EF
from torch.optim import lr_scheduler
from trajGRU import TrajGRU
import numpy as np
from convLSTM import ConvLSTM
from itertools import accumulate

batch_size = cfg.GLOBAL.BATCH_SZIE

IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN

WIDTH = cfg.HKO.ITERATOR.WIDTH
HEIGHT = cfg.HKO.ITERATOR.HEIGHT

STRIDES = cfg.HKO.BENCHMARK.STRIDES
STRIDES_TOTAL = list(accumulate(STRIDES, lambda a, b: a*b))

# print(HEIGHT)
# print(HEIGHT//STRIDES_TOTAL[0])
# print(STRIDES)
# print(STRIDES_TOTAL)

# build model
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, STRIDES[0], 3]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, STRIDES[1], 2]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, STRIDES[2], 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[0], HEIGHT//STRIDES_TOTAL[0]), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[1], HEIGHT//STRIDES_TOTAL[1]), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[2], HEIGHT//STRIDES_TOTAL[2]), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]

forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, STRIDES[2], 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 6, STRIDES[1], 2]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 8, STRIDES[0], 3],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[2], HEIGHT//STRIDES_TOTAL[2]), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[1], HEIGHT//STRIDES_TOTAL[1]), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[0], HEIGHT//STRIDES_TOTAL[0]), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]


# build model
conv2d_params = OrderedDict({
    'conv1_relu_1': [5, 64, 7, STRIDES[0], 1],
    'conv2_relu_1': [64, 192, 5, STRIDES[1], 1],
    'conv3_relu_1': [192, 192, 3, STRIDES[2], 1],
    'deconv1_relu_1': [192, 192, 4, STRIDES[2], 1],
    'deconv2_relu_1': [192, 64, 5, STRIDES[1], 1],
    'deconv3_relu_1': [64, 64, 7, STRIDES[0], 1],
    'conv3_relu_2': [64, 20, 3, 1, 1],
    'conv3_3': [20, 20, 1, 1, 0]
})


# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, STRIDES[0], 3]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, STRIDES[1], 2]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, STRIDES[2], 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[0], HEIGHT//STRIDES_TOTAL[0]),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[1], HEIGHT//STRIDES_TOTAL[1]),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[2], HEIGHT//STRIDES_TOTAL[2]),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, STRIDES[2], 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 6, STRIDES[1], 2]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 8, STRIDES[0], 3],
            'conv3_leaky_2': [8, 8, 3, 2, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[2], HEIGHT//STRIDES_TOTAL[2]),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[1], HEIGHT//STRIDES_TOTAL[1]),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[0], HEIGHT//STRIDES_TOTAL[0]),
                 kernel_size=3, stride=1, padding=1),
    ]
]


convlstm_encoder_params1 = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, STRIDES[0], 3]}),
        OrderedDict({'conv2_leaky_1': [32, 64, 5, STRIDES[1], 2]}),
        OrderedDict({'conv3_leaky_1': [64, 64, 3, STRIDES[2], 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=32, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[0], HEIGHT//STRIDES_TOTAL[0]),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[1], HEIGHT//STRIDES_TOTAL[1]),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[2], HEIGHT//STRIDES_TOTAL[2]),
                 kernel_size=3, stride=1, padding=1),
    ]
]


convlstm_forecaster_params1 = [
    [
        OrderedDict({'deconv1_leaky_1': [64, 64, 4, STRIDES[2], 1]}),
        OrderedDict({'deconv2_leaky_1': [64, 32, 6, STRIDES[1], 2]}),
        OrderedDict({
            'deconv3_leaky_1': [32, 8, 8, STRIDES[0], 3],
            'conv3_leaky_2': [8, 8, 3, 2, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[2], HEIGHT//STRIDES_TOTAL[2]),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[1], HEIGHT//STRIDES_TOTAL[1]),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=32, num_filter=32, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[0], HEIGHT//STRIDES_TOTAL[0]),
                 kernel_size=3, stride=1, padding=1),
    ]
]

def return_params(inner_size):
    convlstm_encoder_params1 = [
        [
            OrderedDict({'conv1_leaky_1': [1, 8, 7, STRIDES[0], 3]}),
            OrderedDict({'conv2_leaky_1': [inner_size//2, inner_size, 5, STRIDES[1], 2]}),
            OrderedDict({'conv3_leaky_1': [inner_size, inner_size, 3, STRIDES[2], 1]}),
        ],

        [
            ConvLSTM(input_channel=8, num_filter=inner_size//2, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[0], HEIGHT//STRIDES_TOTAL[0]),
                    kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=inner_size, num_filter=inner_size, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[1], HEIGHT//STRIDES_TOTAL[1]),
                    kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=inner_size, num_filter=inner_size, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[2], HEIGHT//STRIDES_TOTAL[2]),
                    kernel_size=3, stride=1, padding=1),
        ]
    ]


    convlstm_forecaster_params1 = [
        [
            OrderedDict({'deconv1_leaky_1': [inner_size, inner_size, 4, STRIDES[2], 1]}),
            OrderedDict({'deconv2_leaky_1': [inner_size, inner_size//2, 6, STRIDES[1], 2]}),
            OrderedDict({
                'deconv3_leaky_1': [inner_size//2, 8, 8, STRIDES[0], 3],
                'conv3_leaky_2': [8, 8, 3, 2, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],

        [
            ConvLSTM(input_channel=inner_size, num_filter=inner_size, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[2], HEIGHT//STRIDES_TOTAL[2]),
                    kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=inner_size, num_filter=inner_size, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[1], HEIGHT//STRIDES_TOTAL[1]),
                    kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=inner_size//2, num_filter=inner_size//2, b_h_w=(batch_size, HEIGHT//STRIDES_TOTAL[0], HEIGHT//STRIDES_TOTAL[0]),
                    kernel_size=3, stride=1, padding=1),
        ]
    ]
    return convlstm_encoder_params1, convlstm_forecaster_params1