import torch
import torch.nn as nn
# from unet import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, ToTensor
from einops.layers.torch import Rearrange, Reduce

from einops import rearrange

import numpy as np
# from perceiver_pytorch import MultiPerceiver, Perceiver
# from perceiver_pytorch.modalities import InputModality
# from perceiver_pytorch.encoders import ImageEncoder
# from perceiver_pytorch.decoders import ImageDecoder

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation, bias=False, dilation=dilation),
#             # nn.BatchNorm2d(mid_channels),
#             nn.GroupNorm(4, mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=dilation, bias=False, dilation=dilation),
#             # nn.BatchNorm2d(out_channels),
#             nn.GroupNorm(4, out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels, **args):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 5, stride=2, padding=2),
#             # nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels, **args)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True, **args):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, **args)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels, **args)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class Same(nn.Module):
#     """Double conv with merge."""
#     def __init__(self, in_channels, out_channels, **args):
#         super().__init__()
#         self.down = nn.MaxPool2d(2)
#         self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, **args)

#     def forward(self, x1, x2):
#         x2 = self.down(x2)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1, kernels_per_layer=kernels_per_layer),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1, kernels_per_layer=kernels_per_layer),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvNorm(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation, bias=False, dilation=dilation),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=dilation, bias=False, dilation=dilation),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, reduction_ratio=16, separate=True,**args):
        super().__init__()
        DoubleConv = DoubleConvDS if separate else DoubleConvNorm
        self.maxpool_conv = nn.Sequential(*([nn.Conv2d(in_channels, in_channels, 5, stride=2, padding=2),
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, **args)] + ([CBAM(out_channels, reduction_ratio=reduction_ratio)] if separate else []))
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, separate=True,**args):
        super().__init__()
        DoubleConv = DoubleConvDS if separate else DoubleConvNorm
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, **args)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, **args)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Same(nn.Module):
    """Double conv with merge."""
    def __init__(self, in_channels, out_channels, separate=True, **args):
        super().__init__()
        DoubleConv = DoubleConvDS if separate else DoubleConvNorm
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, **args)

    def forward(self, x1, x2):
        x2 = self.down(x2)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SameMerge(nn.Module):
    """Double conv with merge."""
    def __init__(self, in_channels, out_channels, separate=True, **args):
        super().__init__()
        DoubleConv = DoubleConvDS if separate else DoubleConvNorm
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, **args)

    def forward(self, x1, x2):
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SameNoMerge(nn.Module):
    """Double conv """
    def __init__(self, in_channels, out_channels, reduction_ratio=16, separate=True, **args):
        super().__init__()
        DoubleConv = DoubleConvDS if separate else DoubleConvNorm
        # self.maxpool_conv = nn.Sequential(
        #     DoubleConv(in_channels, out_channels, **args),
        #     CBAM(out_channels, reduction_ratio=reduction_ratio)
        # )
        self.maxpool_conv = nn.Sequential(
            *([DoubleConv(in_channels, out_channels, **args)] + ([CBAM(out_channels, reduction_ratio=reduction_ratio)] if separate else []))
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



##############CONV ATTENTION

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        # self.bn = nn.BatchNorm2d(1)
        self.bn = nn.GroupNorm(1, 1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out
        

#########################################
#       Improve this basic model!       #
#########################################
# SIZE = 128
# SIZE = 128
# MEAN = 333.81436
MEAN = 299.17117
# STD = 152.96744
STD = 146.06215
class Encoder(nn.Module):
    def __init__(self, config, inputs=12, outputs=24, sigmoid=False) -> None:
        super().__init__()
        self.outputs = outputs
        self.inputs = inputs
        self.config = config
        self.normalize = config['normalize']
        self.swap = config['swap']
        self.dropout = nn.Dropout(p=config['dropout'])
        # SIZE = config['size']
        SIZE = config['separate'][0]
        DoubleConv = DoubleConvDS if config['separate'][1] else DoubleConvNorm
        self.input = DoubleConv(inputs, SIZE)
        n_layers = config['n_layers']
        self.encoder = nn.ModuleList([Down(SIZE, SIZE*2, separate=config['separate'][1]), Down(SIZE*2, SIZE*4, separate=config['separate'][1])] + [SameNoMerge(SIZE*4, SIZE*4, separate=config['separate'][1]) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([SameMerge(SIZE*8, SIZE*4, separate=config['separate'][1]) for _ in range(n_layers-1)] + ([SameMerge(SIZE*8, SIZE*2, separate=config['separate'][1])] if n_layers != 0 else []) + [Up(SIZE*4, SIZE, separate=config['separate'][1]), Same(SIZE*2, SIZE, separate=config['separate'][1])])
        # self.decoder = nn.ModuleList([Up(SIZE*2**i, SIZE*2**(i - 2), separate=config['separate'][1]) for i in range(n_layers, 1, -1)] + [Same(SIZE*2, SIZE, separate=config['separate'][1])])
        if sigmoid:
            self.output = nn.Sequential(
                OutConv(SIZE, outputs),
                # nn.ReLU()
                nn.Sigmoid()
            )
        else:
            self.output = OutConv(SIZE, outputs)

    def rencoding(self, features):
        x = self.input(features)
        xs = [x]
        for enc in self.encoder:
            xs.append(enc(xs[-1]))

        if self.swap:
            xs.pop()
            ys = [xs[-1]]
        else:
            ys = [xs[-1]]
            xs.pop()

        for dec in self.decoder:
            ys.append(dec(ys[-1], xs.pop()))
        x = self.output(ys[-1])
        return x
    
    def forward(self, features):
        x = features
        if self.config['local_norm']:
            MEAN = torch.mean(x)
            STD = torch.std(x)
        def preprocessing(x, **args):
            if 'minmax' in self.normalize:
                mi1 = x.min()
                ma1 = x.max()
                x -= mi1
                x /= (ma1 - mi1)
                x *= 2
                x -= 1
                return x, {'mi1':mi1, 'ma1':ma1}
            elif 'standardize' in self.normalize:
                x -= MEAN
                x /= STD
                return x, {}

        def postprocessing(x, **args):
            if 'minmax' in self.normalize:
                x += 1
                x /= 2
                x *= (ma1 - mi1)
                x += mi1
                return x
            elif 'standardize' in self.normalize:
                return x*STD + MEAN

        if self.normalize:
            x, extra = preprocessing(x, MEAN=MEAN, STD=STD)
        x = self.dropout(x)
        x = self.rencoding(x)
        if self.normalize:
            if self.config['output_std'] != 0:
                MEAN = self.config['output_mean']
                STD = self.config['output_std']
            return postprocessing(x, MEAN=MEAN, STD=STD, **extra)
        else:
            return x



class Model(nn.Module):
    def __init__(self, last_n):
        super().__init__()
        self.last_n = last_n
        self.eps = 1e-7
        # self.layer1 = nn.Linear(in_features=last_n, out_features=1)
        self.weight_raw = nn.Parameter(torch.ones(1, last_n))
        # self.layer2 = nn.Linear(in_features=128*128, out_features=1 * 64 * 64)
        # self.layer3 = nn.Linear(in_features=256, out_features=24 * 64 * 64)

    def forward(self, features):
        x = features.transpose(1, 2).transpose(2, 3) #/ 256
        print('PROCESSED', x)
        weight = self.weight_raw / self.weight_raw.sum(1, keepdim=True).clamp(min=self.eps)
        print('WEIGHTS', weight)
        x = F.linear(x, weight)
        x = x.transpose(2, 3).transpose(1, 2)
        # x = torch.relu(self.layer2(x))
        # x = torch.relu(self.layer3(x))
        return x[::, ::, 32:96, 32:96]# * 256


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        # self.processing = nn.BatchNorm2d(4*self.hidden_dim)
        self.processing = nn.GroupNorm(4, 4*self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.processing(self.conv(combined))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(4, out_channels))

    def forward(self, inputs):
        '''

        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)

class Encoder1(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.layers = []
        self.dropout = dropout
        for idx, params in enumerate(encoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            # layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GroupNorm(4 if out_ch % 4 == 0 else 1, out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        elif type == 'conv_output':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        outputs = [x]
        for layer in self.layers:
            if 'conv_' in layer:
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
            x = getattr(self, layer)(x)
            if 'conv_' in layer: x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            if 'convlstm' in layer: outputs.append(x)
        return outputs

class Decoder1(nn.Module):
    def __init__(self, decoder, dropout=0.1):
        super().__init__()
        self.layers = []
        self.dropout = dropout
        for idx, params in enumerate(decoder):
            setattr(self, params[0]+'_'+str(idx), self._make_layer(*params))
            self.layers.append(params[0]+'_'+str(idx))
    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            # layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GroupNorm(4 if out_ch % 4 == 0 else 1, out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
            elif activation == 'sigmoid': layers.append(nn.Sigmoid())
        elif type == 'convoutput':
            # layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        elif type == 'convlstm':
            layers.append(ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride))
        elif type == 'deconv':
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            # layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GroupNorm(4 if out_ch % 4 == 0 else 1, out_ch))
            if activation == 'leaky': layers.append(nn.LeakyReLU(inplace=True))
            elif activation == 'relu': layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, encoder_outputs):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        idx = len(encoder_outputs)-1
        for layer in self.layers:
            if 'conv_' in layer or 'deconv_' in layer:
                x = encoder_outputs[idx]
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
                x = getattr(self, layer)(x)
                x = x.view(B, S, x.shape[1], x.shape[2], x.shape[3])
            elif 'convlstm' in layer:
                idx -= 1
                x = torch.cat([encoder_outputs[idx], x], dim=2)
                x = getattr(self, layer)(x)
                encoder_outputs[idx] = x
            elif 'convoutput' in layer:
                x = encoder_outputs[idx]
                B, S, C, H, W = x.shape
                x = x.view(B*S, C, H, W)
                x = getattr(self, layer)(x)
                x = x.view(B, S*x.shape[1], 1, x.shape[2], x.shape[3])
        return x

class ConvLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dropout = nn.Dropout(p=0.1)
        self.encoder = Encoder1(config['encoder'], dropout=config['dropout'])
        self.decoder = Decoder1(config['decoder'], dropout=config['dropout'])

    def forward(self, x, fac=1023):
        x -= MEAN
        x /= STD
        # x = self.dropout(x)
        x = self.encoder(x)
        x = self.decoder(x)
        # x = x*fac
        return x*STD + MEAN
        # return x#*STD + MEAN

class ModelWrapper(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)



# def video_preprocess(encoder, video):    
#     video1 = rearrange(video, 'b t h w c -> b t c h w')
#     res = encoder(video1)
#     return rearrange(res, 'b c h w -> b (c h) w')

# class PerceiverPredict(nn.Module):
#     def __init__(self, input_size=128, space_down=2) -> None:
#         super().__init__() 
#         self.input_size = input_size
#         self.space_down = space_down
#         max_frequency = 16.0
#         new_size = input_size//space_down
#         self.new_size = new_size
#         video_modality = InputModality(
#             name="timeseries",
#             input_channels=1,
#             input_axis=3,  # number of axes, 3 for video
#             num_freq_bands=new_size,  # number of freq bands, with original value (2 * K + 1)
#             max_freq=max_frequency,  # maximum frequency, hyperparameter depending on how fine the data is, should be Nyquist frequency (i.e. 112 for 224 input image)
#             sin_only=False,  # Whether if sine only for Fourier encoding, TODO test more
#             fourier_encode=True,  # Whether to encode position with Fourier features
#         )
#         self.model = MultiPerceiver(
#             modalities=[video_modality],
#             queries_dim=new_size,
#             depth=6,
#             forecast_steps=12,
#             output_shape=(24,new_size, new_size),
#         )
#         self.preprocessor = ImageEncoder(prep_type='patches', spatial_downsample=space_down, temporal_downsample=12)
#         self.decoder = ImageDecoder(postprocess_type='conv1x1', input_channels=12*space_down**2*new_size, output_channels=1, spatial_upsample=1, temporal_upsample=1)
#         self.output_processing = nn.Sigmoid()

#     def forward(self, features):
#         x = features/256
#         query = video_preprocess(self.preprocessor, x)
#         # print('QUERY', query)
#         inp = {'timeseries': x}
#         out = self.model(inp, queries=query)
#         out = rearrange(out, "b c (t w h) -> b t c h w", t=24, h=self.new_size, w = self.new_size)
#         out = self.decoder(out)
#         # print(out)
#         return self.output_processing(out)*256