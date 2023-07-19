import torch
import torch.nn as nn

from src.ChannelAttention import ChannelAttentionModule
from src.ConvFusion import ConvFusion


class MyModel(nn.Module):
    def __init__(self, up_channel=16):
        super(MyModel, self).__init__()

        inter_channels_ = 4  # 输入4通道
        inter_channels = up_channel  # 升到16通道

        self.conv_fusion = ConvFusion()

        self.conv_c1 = nn.Sequential(
            nn.Conv1d(inter_channels_, inter_channels, 1, padding=0, bias=False),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(True)
        )

        self.cam = ChannelAttentionModule()

        self.conv_c2 = nn.Sequential(
            nn.Conv1d(inter_channels, inter_channels, 1, padding=0, bias=False),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        conv_fusion = self.conv_fusion(x)

        channel_fusion_0 = self.conv_c1(x)  # batch_size * 8 * 2048
        channel_fusion_1 = self.cam(channel_fusion_0)  # batch_size * 8 * 2048
        channel_fusion_2 = self.conv_c2(channel_fusion_1)  # batch_size * 8 * 2048

        return conv_fusion + channel_fusion_2





