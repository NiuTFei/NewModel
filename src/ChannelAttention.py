import torch
from torch import nn


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