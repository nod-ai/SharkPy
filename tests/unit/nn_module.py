from pi import empty
from pi import nn


class MyConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        y = self.conv(x)
        z = y + y
        w = z * z
        return w


def simple_conv2d():
    x = empty((1, 3, 32, 32))
    my_conv = MyConv2d()
    y = my_conv(x)
    return y


simple_conv2d()


class Conv2dNoPaddingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 10, 3, bias=False)

    def forward(self, x):
        return self.conv(x)
