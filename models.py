import torch.nn as nn
import torch




class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, c_in: int, c_internal: int):
        super(SimpleResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_internal, kernel_size=3, padding=1),
            ResBlock(c_internal),
            ResBlock(c_internal),
            nn.Conv2d(c_internal, c_in, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)
    


if __name__ == "__main__":
    print("Testing SimpleResNet...")
    model = SimpleResNet(1, 16)
    x = torch.randn((4, 1, 16, 16))
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)