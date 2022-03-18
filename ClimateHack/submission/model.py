import torch.nn as nn
import torch
from submission.utils import *
from submission.ViT import ViT, ViT_Skip, ViT_encode, ViT_decode, ViT_Temp
from einops import rearrange
#from utils import *
#from metnet import MetNet

class UNET_ConvGRU(nn.Module):
    def __init__(self,in_time_steps,out_time_steps):
        super().__init__()
        """ Encoder """
        self.out_timesteps = out_time_steps
        init_channels = 16
        self.e1 = ConvGRU_block(1, init_channels, in_time_steps)
        self.e2 = ConvGRU_block(init_channels, 2*init_channels, in_time_steps) 
        self.e3 = ConvGRU_block(2*init_channels, 4*init_channels, in_time_steps)
        self.e4 = ConvGRU_block(4*init_channels, 8*init_channels, in_time_steps)
        self.e5 = ConvGRU_block(8*init_channels, 16*init_channels, in_time_steps)
        self.pool = TimeDistributed(nn.MaxPool2d((2,2)))
        '''Bottleneck'''
        #self.bottle = bottle_block(12, 24, kernel_size=1)
        self.skip1 = nn.Conv3d(in_time_steps, out_time_steps, kernel_size=1, padding=0)
        self.skip2 = nn.Conv3d(in_time_steps, out_time_steps, kernel_size=1, padding=0)
        self.skip3 = nn.Conv3d(in_time_steps, out_time_steps, kernel_size=1, padding=0)
        self.skip4 = nn.Conv3d(in_time_steps, out_time_steps, kernel_size=1, padding=0)
        self.skip5 = nn.Conv3d(in_time_steps, out_time_steps, kernel_size=1, padding=0)
        """ Decoder """
        self.d0 = ConvGRU_decode(16*init_channels, 8*init_channels, out_time_steps)
        self.d1 = ConvGRU_decode(8*init_channels, 4*init_channels, out_time_steps)
        self.d2 = ConvGRU_decode(4*init_channels, 2*init_channels, out_time_steps)
        self.d3 = ConvGRU_decode(2*init_channels, init_channels, out_time_steps) # decode blocks include upsample
        self.d4 = ConvGRU_block(2*init_channels, init_channels, out_time_steps)
        """ Classifier """
        self.outputs = TimeDistributed(nn.Conv2d(init_channels, 1, kernel_size=1, padding=0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """ Encoder """
        s1, h1 = self.e1(inputs)
        s2, h2 = self.e2(self.pool(s1)) # out = 64x64
        s3, h3 = self.e3(self.pool(s2)) # 32 x 32
        s4, h4 = self.e4(self.pool(s3)) # 16 x 16
        out, h5 = self.e5(self.pool(s4)) # 8 x 8
        """ Bottleneck """
        #b = self.bottle(out)
        b = torch.zeros((out.shape[0],self.out_timesteps,out.shape[2],out.shape[3],out.shape[4]), dtype=out.dtype, device=out.device)
        """ Decoder """
        dout = self.d0(b, self.skip5(out), h5)
        dout = self.d1(dout, self.skip4(s4), h4)
        dout = self.d2(dout, self.skip3(s3), h3)
        dout = self.d3(dout, self.skip2(s2), h2)
        dout, _ = self.d4(dout, self.skip1(s1), h1)
        """ Classifier """
        dout = self.outputs(dout)
        dout = self.sigmoid(dout)
        return dout

class UNETViT(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        """ Encoder """
        init_channels = 24
        conv_type = None
        #conv_type = "CBAM"
        self.e1 = encoder_block(in_channels, init_channels, (2,2), 64, conv_type)
        self.e2 = encoder_block(init_channels, 2*init_channels, (2,2), 32, conv_type)
        self.e3 = encoder_block(2*init_channels, 4*init_channels, (2,2), 16, conv_type)
        self.e4 = encoder_block(4*init_channels, 8*init_channels, (2,2), 8, conv_type)
        self.e5 = encoder_block(8*init_channels, 16*init_channels, (2,2), 4, conv_type)
        # self.e6 = encoder_block(512, 1024)
        """ Bottleneck """
        #self.vit_temp = ViT_Temp(img_dim=img_dim,
        #               in_channels=2*out_c,  # input features' channels (encoder)
        #               # transformer inside dimension that input features will be projected
        #               # out will be [batch, dim_out_vit_tokens, dim ]
        #               dim=768,
        #               blocks=1,
        #               heads=1,
        #               dim_linear_block=3072,
        #               classification=False)
        self.vit = ViT(img_dim=4,
                       in_channels=16*init_channels,  # input features' channels (encoder)
                       patch_dim=1,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=1024,
                       blocks=1,
                       heads=1,
                       dim_linear_block=3072,
                       classification=False)
        # to project patches back - undoes vit's patchification
        token_dim = 16*init_channels
        self.project_patches_back = nn.Linear(1024, token_dim)
        # upsampling path
        self.b = conv_block(16*init_channels, 32*init_channels)

        #self.skip1 = ViT_Skip(128, init_channels)
        #self.skip2 = ViT_Skip(64, 2*init_channels)
        #self.skip3 = ViT_Skip(32, 4*init_channels)
        #self.skip4 = ViT_Skip(16, 8*init_channels)
        #self.skip5 = ViT_Skip(8, 16*init_channels)

        """ Decoder """
        self.d0 = decoder_block(32*init_channels, 16*init_channels, (2,2), 8, conv_type)
        # self.d0b = decoder_block(1024, 512)
        self.d1 = decoder_block(16*init_channels, 8*init_channels, (2,2), 16, conv_type)
        self.d2 = decoder_block(8*init_channels, 4*init_channels, (2,2), 32, conv_type)
        self.d3 = decoder_block(4*init_channels, 2*init_channels, (2,2), 64, conv_type)
        self.d4 = decoder_block(2*init_channels, init_channels, (2,2), 128, conv_type)
        """ Classifier """
        self.outputs = nn.Conv2d(init_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        #s6, p6 = self.e6(p5)
        """ Bottleneck """
        #b = self.vit_temp(p5)
        b = self.vit(p5)  # out shape of number_of_patches, vit_transformer_dim

        # from [number_of_patches, vit_transformer_dim] -> [number_of_patches, token_dim]
        b = self.project_patches_back(b)

        # from [batch, number_of_patches, token_dim] -> [batch, channels, img_dim_vit, img_dim_vit]
        b = rearrange(b, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                      x=4, y=4,
                      patch_x=1, patch_y=1)
        b = self.b(b)
        """ Decoder """
        d0 = self.d0(b, s5)
        #d0b = self.d0b(d0,s5)
        d1 = self.d1(d0, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        outputs = self.sigmoid(outputs)
        return outputs

# U-NET BLOCKS AND MODEL
class ViT_Net(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        """ Encoder """
        init_channels = 24
        base_heads = 1
        conv_type = None
        #conv_type = "CBAM"
        self.e1 = encoder_block(in_channels, init_channels, (2,2), conv_type)
        self.e2 = ViT_encode(init_channels, 2*init_channels, 64, 2*base_heads)
        self.e3 = ViT_encode(2*init_channels, 4*init_channels, 32, 4*base_heads)
        self.e4 = ViT_encode(4*init_channels, 8*init_channels, 16, 8*base_heads)
        #self.e5 = encoder_block(8*init_channels, 16*init_channels, (2,2), conv_type)
        """ Bottleneck """
        self.b = conv_block(8*init_channels, 16*init_channels)
        """ Decoder """
        #self.d0 = decoder_block(32*init_channels, 16*init_channels, (2,2), conv_type)
        self.d1 = ViT_decode(16*init_channels, 8*init_channels, 16, 8*base_heads)
        self.d2 = ViT_decode(8*init_channels, 4*init_channels, 32, 4*base_heads)
        self.d3 = ViT_decode(4*init_channels, 2*init_channels, 64, 2*base_heads)
        self.d4 = decoder_block(2*init_channels, init_channels, (2,2), conv_type)
        """ Classifier """
        self.outputs = nn.Conv2d(init_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        #s5, p5 = self.e5(p4)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        #d0 = self.d0(b, s5)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        outputs = self.sigmoid(outputs)
        return outputs

# U-NET BLOCKS AND MODEL
class UNET(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        """ Encoder """
        init_channels = 24
        conv_type = None
        #conv_type = "CBAM"
        self.e1 = encoder_block(in_channels, init_channels, (2,2), conv_type)
        self.e2 = encoder_block(init_channels, 2*init_channels, (2,2), conv_type)
        self.e3 = encoder_block(2*init_channels, 4*init_channels, (2,2), conv_type)
        self.e4 = encoder_block(4*init_channels, 8*init_channels, (2,2), conv_type)
        self.e5 = encoder_block(8*init_channels, 16*init_channels, (2,2), conv_type)
        # self.e6 = encoder_block(512, 1024)
        """ Bottleneck """
        self.b = conv_block(16*init_channels, 32*init_channels)
        #self.b = conv_CBAM(16*init_channels, 32*init_channels)
        """ Decoder """
        self.d0 = decoder_block(32*init_channels, 16*init_channels, (2,2), conv_type)
        # self.d0b = decoder_block(1024, 512)
        self.d1 = decoder_block(16*init_channels, 8*init_channels, (2,2), conv_type)
        self.d2 = decoder_block(8*init_channels, 4*init_channels, (2,2), conv_type)
        self.d3 = decoder_block(4*init_channels, 2*init_channels, (2,2), conv_type)
        self.d4 = decoder_block(2*init_channels, init_channels, (2,2), conv_type)
        """ Classifier """
        self.outputs = nn.Conv2d(init_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        #s6, p6 = self.e6(p5)
        """ Bottleneck """
        b = self.b(p5)
        """ Decoder """
        d0 = self.d0(b, s5)
        #d0b = self.d0b(d0,s5)
        d1 = self.d1(d0, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        outputs = self.sigmoid(outputs)
        return outputs

class UNET64(nn.Module):
    def __init__(self,UNET):
        super().__init__()
        self.UNET = UNET
        self.bn = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.out64 = encoder_block(24,24)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inputs):
        x = self.UNET(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        _,x = self.out64(x)
        outputs = self.sigmoid(x)
        return outputs

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

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
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


class ConvLSTMs(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTMs, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_states=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            pass # Hidden_state initialized as previous final hidden state
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class ConvLSTMModel(nn.Module):
    def __init__(self, time_steps, channels, hidden_dim, out_steps):
        super().__init__()
        self.clstm1 = ConvLSTMs(input_dim=channels, hidden_dim=hidden_dim, kernel_size=(5,5), num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm3d(time_steps)
        self.clstm2 = ConvLSTMs(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=(3,3), num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm3d(time_steps)
        self.clstm3 = ConvLSTMs(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=(1,1), num_layers=1, batch_first=True)
        self.maxpool2d = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.conv3d = nn.Conv3d(hidden_dim, out_steps, (3,3,3), padding='same')
        #self.conv3dch = nn.Conv3d(time_steps, 1, (3,3,3), padding='same')



    def forward(self, x):
        outputs, _ = self.clstm1(x)
        outputs = torch.relu(outputs[0])
        outputs = self.bn1(outputs)
        outputs, _ = self.clstm2(outputs)
        outputs = torch.relu(outputs[0])
        outputs = self.bn2(outputs)
        outputs, _ = self.clstm3(outputs)
        outputs = torch.relu(outputs[0])
        # Reduce channels to 1
        #outputs = self.conv3dch(outputs)
        #outputs = torch.relu(outputs)
        # Permute outputs from (b,t,c,h,w) -> (b,c,t,h,w)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.maxpool2d(outputs)
        # get last outputs sequence form outputs:
        outputs = self.conv3d(outputs)
        outputs = torch.sigmoid(outputs)
        # Permute back  (b,c,t,h,w) -> (b,t,c,h,w)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        return outputs

if __name__ == "__main__":
    x = torch.rand((8,12,1,128,128))
    #y = torch.rand((8,1,24,64,64))
    model = UNET_ConvGRU(12,24)
    y = model(x)
    print(y.shape)



    #MSSIM_criterion = MS_SSIM(data_range=1.0, size_average=True, win_size=3, channel=24)
    #model = ConvLSTMModel(time_steps=1,channels=12,hidden_dim=64,out_steps=24)
    #outs = model(x)
    #print(MSSIM_criterion(outs.squeeze(1),y.squeeze(1)))
    #print(outs.shape)