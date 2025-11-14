import torch.nn as nn
import torch
import math




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


def timestep_embedding(t: torch.Tensor, 
                       dim: int,
                       device: torch.device|str = "cpu") -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1e4) * torch.arange(half,device=device) / half)  # 1/ω_k
    args  = t[:, None] * freqs[None]                                # (B, half)
    emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, dim)
    return emb

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimpleResNetWithTimeInput(nn.Module):
    def __init__(self, 
                 c_in: int, 
                 temb_dim: int, 
                 c_internal: int,
                 device: torch.device|str="cpu"):
        super(SimpleResNetWithTimeInput, self).__init__()
        self.temb_dim = temb_dim
        self.conv_in = nn.Conv2d(c_in, c_internal, kernel_size=3, padding=1)
        self.time_mlp = MLP(temb_dim, 256, c_internal)
        self.resblock1 = ResBlock(c_internal)
        self.resblock2 = ResBlock(c_internal)
        #self.resblock3 = ResBlock(c_internal)
        #self.resblock4 = ResBlock(c_internal)
        self.conv_out = nn.Conv2d(c_internal, c_in, kernel_size=3, padding=1)
        self._device = device
        self.to(device)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim, device=self._device))   # B, c_internal
        temb_broadcast = temb[:, :, None, None]  # B, c_internal, 1, 1
        h = self.conv_in(x) + temb_broadcast     # B, c_internal, H, W
        h = self.resblock1(h) + temb_broadcast   # B, c_internal, H, W
        h = self.resblock2(h)                    # B, c_internal, H, W
        #h = self.resblock3(h) + temb_broadcast   # B, c_internal, H, W
        #h = self.resblock4(h)
        out = self.conv_out(h)  # B, c_in, H, W
        return out
    
class SimpleResNet(nn.Module):
    def __init__(self, c_in: int, c_internal: int):
        super(SimpleResNet, self).__init__()
        self.conv_in = nn.Conv2d(c_in, c_internal, kernel_size=3, padding=1)
        self.resblock1 = ResBlock(c_internal)
        self.resblock2 = ResBlock(c_internal)
        self.conv_out = nn.Conv2d(c_internal, c_in, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        h = self.conv_in(x)   # B, c_internal, H, W
        h = self.resblock1(h)
        h = self.resblock2(h)
        out = self.conv_out(h)  # B, c_in, H, W
        return out
    


if __name__ == "__main__":
    print("Testing SimpleResNet...")
    model = SimpleResNet(1, 16)
    x = torch.randn((4, 1, 16, 16))
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)