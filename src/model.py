import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        feat_a = x  # 数据维度 batch_size * 8 * 2048
        feat_a_transpose = x.permute(0, 2, 1)  # 转换维度 batch_size * 2048 * 8
        attention = torch.bmm(feat_a, feat_a_transpose)  # batch_size * 8 * 8

        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(
            attention) - attention  # batch_size * 8 * 8
        attention = self.softmax(attention_new)  # batch_size * 8 * 8

        feat_b = x
        feat_e = torch.bmm(attention, feat_b)  # batch_size * 8 * 2048

        out = self.beta * feat_e + x

        return out


class ConvFusion(nn.Module):
    def __init__(self, input_dim=4, num_init_features=8):
        super(ConvFusion, self).__init__()
        self.cov0_0 = nn.Conv1d(input_dim, num_init_features, kernel_size=32, stride=1, padding=16)
        self.cov0_1 = nn.Conv1d(num_init_features, num_init_features, kernel_size=16, stride=1, padding=7)
        self.cov0_2 = nn.Conv1d(num_init_features, num_init_features, kernel_size=16, stride=1, padding=13, dilation=2)

    def forward(self, x):
        fea_0 = self.cov0_0(x)      # [100, 32, 2049]
        fea_1 = self.cov0_1(fea_0)  # [100, 32, 2048]
        fea_2 = self.cov0_2(fea_0)  # [100, 32, 2045]
        fea_2 = F.pad(fea_2, (1, 2), 'constant', 0)  # [100, 32, 2048]
        fea = torch.cat((fea_1, fea_2), dim=1)      # [100, 64, 2048],第二个维度为2 * num_init_features
        return fea


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


x = torch.randn(100, 4, 2048)
print(x.shape)
net = MyModel()
y = net(x)  # 通道自注意力网络的输出为 batch_size * 16 * 2048 (通道数增加了）
print(y.shape)


