"""cbam.py"""

"""
Ref: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: Add CBAM to raw TransUNet
"""
import torch
import torch.nn as nn
from submission.ConvGRU import ConvGRU
from submission.ViT import MultiHeadCrossAttention, ViT_Temp
def _stack_tups(tuples, stack_dim=1):
    "Stack tuple of tensors along `stack_dim`"
    return tuple(
        torch.stack([t[i] for t in tuples], dim=stack_dim) for i in list(range(len(tuples[0])))
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class TimeDistributed(nn.Module):
    "Applies `module` over `tdim` identically for each step, use `low_mem` to compute one at a time."

    def __init__(self, module, low_mem=False, tdim=1):
        super().__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim

    def forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*tensors, **kwargs)
        else:
            # only support tdim=1
            inp_shape = tensors[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(*[x.view(bs * seq_len, *x.shape[2:]) for x in tensors], **kwargs)
        return self.format_output(out, bs, seq_len)

    def low_mem_forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        seq_len = tensors[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in tensors]
        out = []
        for i in range(seq_len):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        if isinstance(out[0], tuple):
            return _stack_tups(out, stack_dim=self.tdim)
        return torch.stack(out, dim=self.tdim)

    def format_output(self, out, bs, seq_len):
        "unstack from batchsize outputs"
        if isinstance(out, tuple):
            return tuple(out_i.view(bs, seq_len, *out_i.shape[1:]) for out_i in out)
        return out.view(bs, seq_len, *out.shape[1:])

    def __repr__(self):
        return f"TimeDistributed({self.module})"

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=12):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
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
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=12, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, k_size, pad, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, groups=in_channels, kernel_size=k_size, padding=pad)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class conv_CBAM(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        convtype = nn.Conv2d
        self.residual = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)
        self.conv1 = DepthwiseSeparableConv(in_c, out_c, 3, 1)
        self.bn1 = nn.GroupNorm(num_groups=12, num_channels=out_c, eps=1e-6)
        self.conv2 = DepthwiseSeparableConv(out_c, out_c, 3, 1)
        self.bn2 = nn.GroupNorm(num_groups=12, num_channels=out_c, eps=1e-6)
        self.conv3 = DepthwiseSeparableConv(out_c, out_c, 3, 1)
        self.bn3 = nn.GroupNorm(num_groups=12, num_channels=out_c, eps=1e-6)
        self.cbam1 = CBAM(out_c, kernel_size=7)
        self.cbam2 = CBAM(out_c, kernel_size=3)
        self.bnout = nn.GroupNorm(num_groups=12, num_channels=out_c, eps=1e-6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.0)
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.dropout(self.bn1(x))
        # Second Conv layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(self.bn2(x))
        # Third Conv layer
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(self.bn3(x))
        # Out Conv Layer
        x = self.cbam1(x)
        x = self.cbam2(x)
        # Add residual connection
        x = x + self.residual(inputs)
        x = self.relu(x)
        x = self.dropout(self.bnout(x))
        return x

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        convtype = nn.Conv2d
        self.residual = convtype(in_c, out_c, kernel_size=1, stride=1)
        self.conv1 = convtype(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=12, num_channels=out_c, eps=1e-6)
        self.conv2 = convtype(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=12, num_channels=out_c, eps=1e-6)
        self.conv3 = convtype(out_c, out_c, kernel_size=3, padding=1)
        self.bn3 = nn.GroupNorm(num_groups=12, num_channels=out_c, eps=1e-6)
        self.convout = convtype(out_c, out_c, kernel_size=3, padding=1)
        self.bnout = nn.GroupNorm(num_groups=12, num_channels=out_c, eps=1e-6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.0)
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.dropout(self.bn1(x))
        # Second Conv layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(self.bn2(x))
        # Third Conv layer
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(self.bn3(x))
        # Out Conv Layer
        x = self.convout(x)
        # Add residual connection
        x = x + self.residual(inputs)
        x = self.relu(x)
        x = self.dropout(self.bnout(x))
        return x

class bottle_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.bottle = nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(24)
        self.dropout = nn.Dropout3d(p=0.2)
    def forward(self, inputs):
        x = self.relu(self.bottle(inputs))
        x = self.dropout(self.bn(x))
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, p_size, img_dim, conv_type=None):
        super().__init__()
        if conv_type == "CBAM":
            self.conv = conv_CBAM(in_c, out_c)
        else:
            self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(p_size)
        #self.pool = nn.Conv2d(out_c, out_c, kernel_size=(2,2), stride=(2,2), padding=0)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        #p = self.vit(p)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, p_size, img_dim, conv_type=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=p_size, stride=p_size, padding=0)
        #self.bich = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        #self.biup = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.cross_attend = MultiHeadCrossAttention(in_c, out_c)
        #self.gn = nn.GroupNorm(num_groups=12, num_channels=out_c, eps=1e-6)
        #self.relu = nn.ReLU()
        if conv_type == "CBAM":
            self.conv = conv_CBAM(2*out_c, out_c)
        else:
            self.conv = conv_block(2*out_c, out_c)
    def forward(self, inputs, skip):
        #skipx = self.cross_attend(inputs,skip)
        x = self.up(inputs)
        #x = x + self.bich(self.biup(inputs))
        #x = self.gn(self.relu(x))
        x = torch.cat([x, skip], axis=1)
        #x = self.vit(x)
        x = self.conv(x)
        return x




class convres(nn.Module):
    def __init__(self, in_c, out_c, time_steps):
        super().__init__()
        self.residual = TimeDistributed(nn.Conv2d(in_c, out_c, kernel_size=1, stride=1))
        self.conv1 = TimeDistributed(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1))
        #self.bn1 = nn.BatchNorm2d(out_c)
        self.bn1 = TimeDistributed(nn.GroupNorm(num_groups=12, num_channels=time_steps, eps=1e-6), tdim=2)
        self.conv2 = TimeDistributed(nn.Conv2d(out_c, out_c, kernel_size=3, padding=1))
        #self.bn2 = nn.BatchNorm2d(out_c)
        self.bn2 = TimeDistributed(nn.GroupNorm(num_groups=12, num_channels=time_steps, eps=1e-6), tdim=2)
        self.relu = nn.ReLU()
        self.dropout = TimeDistributed(nn.Dropout2d(p=0.3))
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.dropout(self.bn1(x))
        x = self.conv2(x)
        x = x + self.residual(inputs)
        x = self.relu(x)
        x = self.dropout(self.bn2(x))
        return x

class ConvGRU_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps):
        super().__init__()
        self.resblock = convres(in_c, out_c, time_steps)
        self.ConvGRU = ConvGRU(out_c, out_c, kernel_size=(3,3), n_layers=1, input_p=0.0, hidden_p=0.2, batchnorm=False)
    def forward(self, inputs, skip=None, hid_init=None):
        # Residual Conv Block
        x = inputs
        if torch.is_tensor(skip):
            x = torch.cat([x, skip], axis=2)
        x = self.resblock(x)
        # ConvGRU block
        layer_output, last_state_list = self.ConvGRU(x, hid_init)
        return layer_output, last_state_list

class ConvGRU_decode(nn.Module):
    def __init__(self, in_c, out_c, time_steps):
        super().__init__()
        self.resblock = convres(2*in_c, in_c, time_steps)
        self.ConvGRU = ConvGRU(in_c, in_c, kernel_size=(3,3), n_layers=1, input_p=0.0, hidden_p=0.1, batchnorm=False)
        self.up = TimeDistributed(nn.ConvTranspose2d(in_c, out_c, kernel_size=(2,2), stride=(2,2), padding=0))
    def forward(self, inputs, skip, hid_init=None):
        # Residual Conv Block
        x = torch.cat([inputs, skip], axis=2)
        x = self.resblock(x)
        # ConvGRU block
        x, _ = self.ConvGRU(x, hid_init)
        x = self.up(x)
        return x

