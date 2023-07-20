import torch
from torch import nn
import torch.nn.functional as F


# 编码器
class Q_net(nn.Module):
    def __init__(self, input_dim=1, num_init_features=8, output_dim=1):
        super(Q_net, self).__init__()

        self.feature = nn.Sequential(
            nn.BatchNorm1d(num_init_features * 2),
            nn.Mish(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),

            nn.Conv1d(num_init_features * 2, num_init_features * 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_init_features * 2),
            nn.Mish(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),

            nn.Conv1d(num_init_features * 2, num_init_features * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_init_features * 2),
            nn.Mish(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),

            nn.Conv1d(num_init_features * 2, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_init_features),
            nn.Mish(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False)
        )
        self.conFC1 = nn.Conv1d(num_init_features, 500, kernel_size=128, stride=1, bias=False)
        self.mishFC1 = nn.Mish(inplace=True)
        self.conFC2 = nn.Conv1d(500, 100, kernel_size=1, stride=1, bias=False)
        self.mishFC2 = nn.Mish(inplace=True)
        self.conFC3 = nn.Conv1d(100, output_dim, kernel_size=1, stride=1, bias=False)
        self.mishFC3 = nn.Mish(inplace=True)

    def forward(self, x):
        # x: batch_size * 16 * 2048

        x = self.feature(x)  # channel 8
        x = self.conFC1(x)  # channel 500
        x = F.dropout(x)
        x = self.mishFC1(x)
        x = self.conFC2(x)  # channel 100
        x = F.dropout(x)
        x = self.mishFC2(x)
        x = self.conFC3(x)  # channel 1
        x = F.dropout(x)
        x = self.mishFC3(x)
        x_flatten = torch.flatten(x, 1)
        return x, x_flatten


"""
编码器的输入维度num_init_features，是融合卷积的输出的一半，也是通道自注意力输出通道的一半
隐变量的维度为编码器中的output_dim, 解码器中的input_dim
"""


# 解码器
class P_net(nn.Module):
    def __init__(self, num_init_features=8, input_dim=1, output_dim=1):
        super(P_net, self).__init__()
        self.covFC3 = nn.Conv1d(input_dim, 100, kernel_size=1, stride=1, bias=False)
        self.mishFC3 = nn.Mish(inplace=True)
        self.covFC2 = nn.ConvTranspose1d(100, 500, kernel_size=1, stride=1, bias=False)
        self.mishFC2 = nn.Mish(inplace=True)
        self.covFC1 = nn.ConvTranspose1d(500, num_init_features, kernel_size=128, stride=1, bias=False)
        self.mishFC1 = nn.Mish(inplace=True)
        self.feature = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(num_init_features, num_init_features * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_init_features * 2),
            nn.Mish(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(num_init_features * 2, num_init_features * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_init_features * 2),
            nn.Mish(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(num_init_features * 2, num_init_features * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_init_features * 2),
            nn.Mish(inplace=True)
        )
        self.upsample_1 = nn.Upsample(scale_factor=2)

        self.lastcov = nn.ConvTranspose1d(num_init_features, output_dim, kernel_size=32, stride=1, padding=16,
                                          bias=False)

    def forward(self, x):
        x = self.covFC3(x)      # channel 100
        x = F.dropout(x)
        x = self.mishFC3(x)
        x = self.covFC2(x)      # channel 500
        x = F.dropout(x)
        x = self.mishFC2(x)
        x = self.covFC1(x)      # channel 8
        x = F.dropout(x)
        x = self.mishFC1(x)

        x = self.feature(x)     # channel 16
        signal = self.upsample_1(x)  # channel 16

        signal_flatten = torch.flatten(signal, 1)
        return signal, signal_flatten


# 对抗网络
class D_net_gauss(nn.Module):
    def __init__(self, z_dim=2, N=1000):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
        self.mish = nn.Mish(inplace=True)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x)
        x = self.mish(x)
        x = self.lin2(x)
        x = F.dropout(x)
        x = self.mish(x)
        out = self.lin3(x)
        return out


Q = Q_net()
print(sum(p.numel() for p in Q.parameters()))
