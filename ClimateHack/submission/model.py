import torch.nn as nn
import torch
import sys
sys.path.insert(1, './submission')
from utils import *
from ViT import ViT, ViT_Skip, ViT_encode, ViT_decode, ViT_Temp
from einops import rearrange
from Focal_Transformer.classification.focal_transformer_v2 import FocalTransformer, FocalTransformerSys

# Focal Transformer: https://github.com/microsoft/Focal-Transformer
# Uses focal transformer as both encoder and decoder in UNET like fashion. Inspired by Swin-UNet
class Focal_FullUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FocalTransformerSys(img_size=128, patch_size=2, in_chans=12, embed_dim=48, depths=[2,2,6,2,1], drop_path_rate=0.2, 
            focal_stages=[0, 1, 2, 3, 4], focal_levels=[2,2,2,2,2], expand_sizes=[3,3,3,3,3], expand_layer="all", 
            num_heads=[3,6,12,24,24],
            focal_windows=[7,5,3,1,1], 
            window_size=8,
            use_conv_embed=True, 
            use_shift=False, 
        )
        """ Classifier """
        self.outputs = nn.Conv2d(48, 24, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """ Encoder """
        x = self.model(inputs)
        x = rearrange(x, 'b (x y) c -> b c x y', x=64, y=64)
        """ Classifier """
        outputs = self.outputs(x)
        outputs = self.sigmoid(outputs)
        return outputs

# Uses focal transformer as encoder and UNET convolutional decoder with convolutional bottleneck.
class Focal_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        embed = 48
        conv_type = "CBAM"
        self.encoder = FocalTransformer(img_size=128, patch_size=2, in_chans=12, embed_dim=embed, depths=[2,2,6,2], drop_path_rate=0.2, 
            focal_levels=[2,2,2,2], expand_sizes=[3,3,3,3], expand_layer="all", 
            num_heads=[3,6,12,24],
            focal_windows=[7,5,3,1], 
            window_size=8,
            use_conv_embed=True, 
            use_shift=False, 
        )
        self.resid = conv_CBAM(12,embed)

        self.pool = nn.MaxPool2d((2,2))
        self.bottle = conv_CBAM(8*embed, 16*embed)
        self.bich = nn.Conv2d(8*embed, 8*embed, kernel_size=3, padding=1, padding_mode='reflect')
        self.biup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.d1 = decoder_block(16*embed, 8*embed, (2,2), 16, False, conv_type)
        self.d2 = decoder_block(8*embed, 4*embed, (2,2), 32, False, conv_type)
        self.d3 = decoder_block(4*embed, 2*embed, (2,2), 64, False, conv_type)
        self.d4 = decoder_block(2*embed, embed, (2,2), 128, False, conv_type)

        """ Classifier """
        self.outputs = nn.Conv2d(embed, 24, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """ Encoder """
        x, skips = self.encoder(inputs)

        """ Bottleneck """
        p = self.pool(x)
        p = self.bottle(p)

        """ Decoder """
        d1 = self.d1(p, x)
        d2 = self.d2(d1, skips[2])
        d3 = self.d3(d2, skips[1])
        d4 = self.d4(d3, skips[0])
 
        """ Classifier """
        outputs = self.outputs(d4)
        outputs = self.sigmoid(outputs)
        return outputs

# Implementation of ConvGRU encoder decoder from: https://github.com/jleinonen/weather4cast-bigdata
# final hidden states from ConvGRUs are used as initial hidden states for decoder ConvGRU's (similar to Unet skip connection)
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

#### FINAL SUBMISSION ARCHITECTURE FOR University of Toronto. Test MS_SSIM Score: 0.83602  ####
# 5-level UNet augmented with Vision Transformer in bottleneck
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

        """ Vision Transformer Bottleneck """
        self.vit = ViT(img_dim=4,
                       in_channels=16*init_channels,  # input features' channels (encoder)
                       out_channels=32*init_channels,
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

        """ Decoder """
        self.d0 = decoder_block(32*init_channels, 16*init_channels, (2,2), 8, True, conv_type)
        # self.d0b = decoder_block(1024, 512)
        self.d1 = decoder_block(16*init_channels, 8*init_channels, (2,2), 16, True, conv_type)
        self.d2 = decoder_block(8*init_channels, 4*init_channels, (2,2), 32, True, conv_type)
        self.d3 = decoder_block(4*init_channels, 2*init_channels, (2,2), 64, True, conv_type)
        self.d4 = decoder_block(2*init_channels, init_channels, (2,2), 128, True, conv_type)

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

        """ Bottleneck """
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
        d1 = self.d1(d0, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)
        outputs = self.sigmoid(outputs)
        return outputs

# Use vision transformer for each level of UNet encoder and decoder. This was created before we learned about multi-resolution vision transformers.
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

# Normal Convolutional UNet
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