from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Create a dataset of n points in k-dimensional space
# Each point is sampled uniformly from [start, end]^k
class KDBoxDataset(Dataset):
    def __init__(self, n: int, k: int, start: float, end: float): 
        self.n = n
        self.points = (torch.rand(n, k) * (end - start)) + start
    def __len__(self): return self.n
    def __getitem__(self, i):
        return self.points[i]


# Create a dataset of images with K lines
# Each line is defined by a direction (up-down or left-right) and a position
class PerpendicularLinesDataset(Dataset):
    def __init__(self, n: int, shape: Tuple[int, int,int], k: int):
        self.shape = shape
        self.n = n
        self.k = k
        self.images = torch.zeros((n, *shape))
        assert shape[0] == 1

        for i in range(n):
            img = torch.zeros(shape)

            for _ in range(k):
                direction = torch.randint(0, 2, (1,)).item()  # 0: vertical, 1: horizontal
                position = torch.randint(0, shape[1+direction], (1,)).item()

                #print(f"D={direction}  P={position}")
                if direction == 0:  # vertical line
                    img[0, :, position] = 1.0
                else:  # horizontal line
                    img[0, position, :] = 1.0
                #print(img[0])
            
            self.images[i] = img
                

    def __len__(self): return self.n

    def __getitem__(self, i):
        return self.images[i]


if __name__ == "__main__":
    print("Testing PerpendicularLinesDataset...")
    ds = PerpendicularLinesDataset(1000, (1, 8, 8), 5)
    #dl = DataLoader(ds, batch_size=32, shuffle=True)

    print (ds.images.shape)
    idx = torch.randint(0, len(ds), (64,))
    images = ds.images[idx]
    print(images[0])
    print (images.shape)
    grid = vutils.make_grid(images, nrow=8, normalize=True, pad_value=1.0)
    print("Grid shape:", grid.shape)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()