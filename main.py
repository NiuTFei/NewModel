import torch

from src.model import MyModel


def test():
    # Use a breakpoint in the code line below to debug your script.
    x = torch.randn(100, 4, 2048)  # 输入信号shape：batch_size * 4 * 2048
    print(x.shape)
    net = MyModel()
    y = net(x)  # 通道自注意力网络的输出为 batch_size * 16 * 2048 (通道数增加了）
    print(y.shape)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
