import torch
import torch.nn as nn
import torch.nn.functional as F

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