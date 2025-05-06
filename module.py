import torch
from torch import nn
import torch.nn.functional as F

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, use_se=False):
        super(RepConv2d, self).__init__()

        self.in_channels = in_channels

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.PReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

        self.dropout = nn.Dropout2d(0.0)

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        out = self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
        out = self.dropout(out)
        return out


import numpy as np
import torch
from torch import nn
from torch.nn import init


# 其他注意力机制
class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out+x
        return out

class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out


# Source from https://github.com/haoshao-nku/medical_seg
# 提出的注意力的实现模块
class MSCAAttention(nn.Module):
    """Multi-Scale Convolutional Attention(MSCA)模块.
    多尺度特征提取：通过多个卷积核大小和填充的卷积操作，以提取不同尺度的特征信息。
                这些卷积操作包括一个具有较大卷积核的初始卷积 (self.conv0) 和多个后续的卷积操作（self.conv0_1，self.conv0_2，self.conv1_1，self.conv1_2，self.conv2_1，self.conv2_2），每个都针对不同的核大小和填充。
    通道混合：在提取多尺度特征之后，通过对这些特征进行通道混合来整合不同尺度的信息。通道混合操作由最后一个卷积层 self.conv3 完成。
    卷积注意力：最后，通过将通道混合后的特征与输入特征进行逐元素乘法，实现了一种卷积注意力机制。这意味着模块通过对不同通道的特征赋予不同的权重来选择性地强调或抑制输入特征。
    """
    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        """

        :param channels: 通道数.
        :param kernel_sizes: 注意力核大小. 默认: [5, [1, 7], [1, 11], [1, 21]].
        :param paddings: 注意力模块中相应填充值的个数.
            默认: [2, [0, 3], [0, 5], [0, 10]].
        """
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()

        attn = self.conv0(x)

        # 多尺度特征提取
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        # 通道融合（也是通过1x1卷积）
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x

class MSCASpatialAttention(nn.Module):
    """
    Spatial Attention Module in Multi-Scale Convolutional Attention Module，多尺度卷积注意力模块中的空间注意模块
    先过1x1卷积，gelu激活后过注意力，再过一次1x1卷积，最后和跳跃连接
    """

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU')):
        """

        :param in_channels: 通道数.
        :param attention_kernel_sizes (list): 注意力核大小. 默认: [5, [1, 7], [1, 11], [1, 21]].
        :param attention_kernel_paddings (list): 注意力模块中相应填充值的个数.
        :param act_cfg (list): 注意力模块中相应填充值的个数.
        """
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # 跳跃连接
        shorcut = x.clone()
        # 先过1x1卷积
        x = self.proj_1(x)
        # 激活
        x = self.activation(x)
        # 过MSCAAttention
        x = self.spatial_gating_unit(x)
        # 1x1卷积
        x = self.proj_2(x)
        # 残差融合
        x = x + shorcut
        return x


"CBAM: Convolutional Block Attention Module "


class HAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((None, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((None, 1))
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,H,1)
        avg_result = self.avgpool(x)  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,H,1)
        max_out = self.se(max_result)  # 共享同一个MLP: (B,C,H,1)--> (B,C,H,1)
        avg_out = self.se(avg_result)  # 共享同一个MLP: (B,C,H,1)--> (B,C,H,1)
        output = self.sigmoid(max_out + avg_out)  # 相加,然后通过sigmoid获得权重:(B,C,H,1)
        return output


class WAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, None))
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,W)
        avg_result = self.avgpool(x)  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,W)
        max_out = self.se(max_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,W)
        avg_out = self.se(avg_result)  # 共享同一个MLP: (B,C,1,1)--> (B,C,1,W)
        output = self.sigmoid(max_out + avg_out)  # 相加,然后通过sigmoid获得权重:(B,C,1,W)
        return output


class CAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:(B,C,H,W)
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 最大值和对应的索引.
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 平均值
        result = torch.cat([max_result, avg_result], 1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        output = self.conv(result)  # 然后重新降维为1维:(B,1,H,W)
        output = self.sigmoid(output)  # 通过sigmoid获得权重:(B,1,H,W)
        return output


class HWCBlock(nn.Module):
    # channel修改512修改为64
    def __init__(self, channel=64, reduction=16, kernel_size=3):
        super().__init__()
        self.HAttention = HAttention(channel=channel, reduction=reduction)
        self.WAttention = WAttention(channel=channel, reduction=reduction)
        self.CAttention = CAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        residual = x
        B, C, H, W = x.size()
        out = x * self.WAttention(x) * self.HAttention(
            x)  # 将输入与通道注意力权重相乘: (B,C,H,W) * (B,C,1,W) * (B,C,H,1) = (B,C,H,W)
        out = out * self.CAttention(out)  # 将更新后的输入与空间注意力权重相乘:(B,C,H,W) * (B,1,H,W) = (B,C,H,W)
        return out + residual