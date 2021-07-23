import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import os

class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.elu = nn.ELU(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.elu(x)
        return x
        
class UpConv(nn.Module):
    """ Up-Sample Section """
    def __init__(self, in_channels, out_channels, kernel_size, scale):
        super().__init__()
        self.scale = scale
        self.conv = ConvBN(in_channels, out_channels, kernel_size)
    
    def forward(self, inputs):
        # Up-Sample used imterpolate op
        x = F.interpolate(inputs, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)

def conv3x3(in_planes:int, out_planes:int, strides:int=1, dilation:int=1):
    """ 3x3 Conv """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strides, padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 Conv"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=1, bias=False)

def gelu():
    return nn.GELU()

def batchnorm(planes:int):
    return nn.BatchNorm2d(planes)

def layernorm(shape):
    return nn.LayerNorm(shape)

def patch_aggr(in_planes, out_planes, kernel_size=2, stride=2):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride)

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer = None,
        # activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if activation_layer is None:
        #     activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            # activation_layer(inplace=True)
        )
        self.out_channels = out_planes

ConvBN = ConvBNActivation

def dw_conv(channels, stride=1, kernel_size=3, norm_layer=None):
    return ConvBN(in_planes=channels, out_planes=channels, kernel_size=kernel_size, stride=stride, norm_layer=norm_layer)


class Local_Perception_Unit(nn.Module):
    def __init__(
        self,
        in_channel,
        stride=1,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        self.dw_conv = dw_conv(channels=in_channel, stride=stride, norm_layer=norm_layer)
    
    def forward(self, inputs):
        B, C, H, W = inputs.shape
        x = self.dw_conv(inputs)
        x = x + inputs
        return x.permute(0, 2, 3, 1)


class MutilHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super().__init__()    
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        B, N, C = query.shape
        query = query.reshape(B, self.dim, self.num_heads, -1)
        key = key.reshape(B, self.dim, self.num_heads, -1)
        value = value.reshape(B, self.dim, self.num_heads, -1)


        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / N**.5 
        attn = F.softmax(scores, dim=-1)
        x = torch.einsum('bhnm,bdhm->bdhn', attn, value)

        x = self.merge(x.contiguous().view(B, self.dim*self.num_heads, -1))

        return x

class LightweightAttn(nn.Module):
    def __init__(
        self,
        in_channel:int,
        k:int,
        num_heads:int
    ):
        super().__init__()
        self.dw_conv1 = dw_conv(channels=in_channel, stride=k, kernel_size=k)
        self.dw_conv2 = dw_conv(channels=in_channel, stride=k, kernel_size=k)

        self.mhsa = MutilHeadSelfAttention(in_channel, num_heads)

        self.linear1 = nn.Linear(in_channel, in_channel)
        self.linear2 = nn.Linear(in_channel, in_channel) 
        self.linear3 = nn.Linear(in_channel, in_channel) 
    
    def forward(self, inputs):
        B, H, W, C = inputs.shape
        inputs_channel_first = inputs.permute(0, 3, 1, 2)
        x0 = self.linear1(inputs) #[Hi × Wi × Ci]
        x1 = self.dw_conv1(inputs_channel_first) #[Ci x Hi/k × Wi/k]
        x2 = self.dw_conv2(inputs_channel_first) #[Ci x Hi/k × Wi/k]

        x0 = x0.reshape(B, -1, C) 
        x1 = x1.flatten(2).permute(0, 2, 1) #[Hi/k*Wi/k x Ci]
        x2 = x2.flatten(2).permute(0, 2, 1) #[Hi/k*Wi/k x Ci]

        x = self.mhsa(x0, x1, x2).reshape(B, H, W, C) #[B, C, H, W]
        
        return x

class InvertedResidualFFN(nn.Module):
    def __init__(
        self,
        in_channel,
        expansion_ratio:int=4
    ):
        super().__init__()
        self.conv1 = conv1x1(in_channel, in_channel)
        self.conv2 = conv1x1(in_channel, in_channel)
        self.gelu1 = gelu()
        self.bn1 = batchnorm(in_channel)

        self.gelu2 = gelu()
        self.bn2 = batchnorm(in_channel)

        self.dw_conv = dw_conv(in_channel)
    
    def forward(self, inputs):
        B, H, W, C = inputs.shape
        inputs_channel_first = inputs.permute(0, 3, 1, 2)

        x = self.conv1(inputs_channel_first)
        out1 = self.bn1(self.gelu1(x))
        out2 = self.dw_conv(out1)
        out = out1 + out2
        out = self.gelu2(out)
        out = self.bn2(self.conv2(out))

        return out.permute(0, 2, 3, 1) 

class Stem(nn.Module):
    def __init__(
        self,
        in_channel=3,
        stem_out_channel:list=[32, 32, 32]
    ):
        super().__init__()
        out_channels = stem_out_channel
        self.conv1 = conv3x3(in_channel, out_channels[0], strides=2)
        self.gelu1 = gelu()
        self.bn1 = batchnorm(out_channels[0])

        self.conv2 = conv3x3(out_channels[0], out_channels[1])
        self.gelu2 = gelu()
        self.bn2 = batchnorm(out_channels[1])

        self.conv3 = conv3x3(out_channels[1], out_channels[2])
        self.gelu3 = gelu()
        self.bn3 = batchnorm(out_channels[2])

    def forward(self, inputs):
        x = self.bn1(self.gelu1(self.conv1(inputs)))
        x = self.bn2(self.gelu2(self.conv2(x)))
        x = self.bn3(self.gelu3(self.conv3(x)))

        return x

class CMT_Block(nn.Module):
    def __init__(
        self,
        in_channel:int,
        k:int,
        num_heads:int,
        stride:int=1,
        norm_layeer=nn.BatchNorm2d
    ):
        super().__init__()
        self.lpu = Local_Perception_Unit(in_channel, stride=stride, norm_layer=norm_layeer)
        self.layer_norm1 = layernorm(in_channel) # [128]
        self.layer_norm2 = layernorm(in_channel)
        self.lmhsa = LightweightAttn(in_channel, k, num_heads)
        self.irffn = InvertedResidualFFN(in_channel)
    
    def forward(self, inputs):
        x0 = self.lpu(inputs)
        x = self.layer_norm1(x0)
        x = self.lmhsa(x)
        res0 = x + x0
        x = self.layer_norm2(res0)
        x = self.irffn(x) #[H, W, C]
        x = res0 + x
        return x.permute(0, 3, 1, 2) # [B, C, H, W]

class CMT(nn.Module):
    def __init__(
        self,
        in_channel:int=3,
        block_nums:list=[4, 4, 20, 4],
        k_blocks:list=[8, 4, 2, 1],
        heads_blocks:list=[1, 2, 4, 8],
        stem_out_channels:list=[32, 32, 32],
        patch_agg_channels:list=[64, 128, 256, 512],
        num_classes:int=1000,
        include_top:bool=False
    ):
        super().__init__()
        self.include_top = include_top
        self.stem = Stem(in_channel, stem_out_channels)
        self.patch_agg1 = patch_aggr(stem_out_channels[-1], patch_agg_channels[0])
        self.patch_agg2 = patch_aggr(patch_agg_channels[0], patch_agg_channels[1])
        self.patch_agg3 = patch_aggr(patch_agg_channels[1], patch_agg_channels[2])
        self.patch_agg4 = patch_aggr(patch_agg_channels[2], patch_agg_channels[3])

        self.stage1 = nn.Sequential(*[CMT_Block(patch_agg_channels[0], k_blocks[0], heads_blocks[0]) for _ in range(block_nums[0])])
        self.stage2 = nn.Sequential(*[CMT_Block(patch_agg_channels[1], k_blocks[1], heads_blocks[1]) for _ in range(block_nums[1])])
        self.stage3 = nn.Sequential(*[CMT_Block(patch_agg_channels[2], k_blocks[2], heads_blocks[2]) for _ in range(block_nums[2])])
        self.stage4 = nn.Sequential(*[CMT_Block(patch_agg_channels[3], k_blocks[3], heads_blocks[3]) for _ in range(block_nums[3])])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(patch_agg_channels[-1], 1280)
        self.head = nn.Linear(1280, num_classes)
    
    def forward(self, inputs):
        x = self.stem(inputs) # [1, 32, 240, 320]

        x = self.patch_agg1(x) # [1, 64, 120, 160]
        x = self.stage1(x) # [1, 64, 120, 160]

        x = self.patch_agg2(x) # [1, 128, 60, 80]
        x = self.stage2(x) # [1, 128, 60, 80]

        x = self.patch_agg3(x) # [1, 256, 30, 40]
        x = self.stage3(x) # [1, 256, 30, 40]

        x = self.patch_agg4(x) # [1, 512, 15, 20]
        x = self.stage4(x) # [1, 512, 15, 20]

        if not self.include_top:
            return x
        else:
            x = self.avgpool(x.flatten(2))
            x = torch.flatten(x, 1)
            x = self.head(self.fc(x))
            return x

class CMT_backbone(nn.Module):
    def __init__(self, coarse_dim, fine_dim):
        super().__init__()
        channels = [32, 64, 128]
        self.cmt = CMT()
        self.stem = self.cmt.stem
        self.patch_agg1 = self.cmt.patch_agg1 
        self.patch_agg2 = self.cmt.patch_agg2 

        self.stage1 = self.cmt.stage1
        self.stage2 = self.cmt.stage2

        # Decoder
        self.upconv3 = UpConv(channels[2], 512, 3, 2)
        self.iconv3 = ConvBN(channels[1]+512, 512, 3, 1)
        self.upconv2 = UpConv(512, 256, 3, 2)
        self.iconv2 = ConvBN(channels[0]+256, 256, 3, 1)
        # coarse conv
        self.conv_coarse = ConvBN(channels[2], coarse_dim, 1, 1)
        # fine conv
        self.conv_fine = ConvBN(256, fine_dim, 1, 1)

    def skipconnect(self, x1, x2):
        diff_x = x2.size()[2] - x1.size()[2] # difference of h
        diff_y = x1.size()[3] - x1.size()[3] # difference of w
        x1 = F.pad(x1, (diff_x//2, diff_x-diff_x//2,
                        diff_y//2, diff_y-diff_y//2))
        x = torch.cat([x2, x1], dim=1) # dim
        return x

    def forward(self, inputs):
        x1 = self.stem(inputs) # [1, 32, 240, 320]

        x2 = self.patch_agg1(x1) # [1, 76, 120, 160]
        x2 = self.stage1(x2) # [1, 76, 120, 160]

        x3 = self.patch_agg2(x2) # [1, 156, 60, 80]
        x3 = self.stage2(x3)

        coarse_feature = self.conv_coarse(x3) 

        x = self.upconv3(x3) # 512-d [1, 512, 120, 160]
        x = self.skipconnect(x2, x) # 640-d [1, 640, 120, 160]
        x = self.iconv3(x) # 512-d [1, 512, 120, 160]
        x = self.upconv2(x) # 256-d [1, 256, 240, 320]
        x = self.skipconnect(x1, x)
        x = self.iconv2(x) # 256-d [1, 256, 240, 320]

        fine_feature = self.conv_fine(x)

        return [coarse_feature, fine_feature]

        



