import torch.nn as nn
import torch
import datasets as mydatasets
import models
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from diffusers import UNet2DModel
import math
from pathlib import Path

def cosine_abar_to_beta(T: int, 
                        s=0.008, 
                        device="cpu", 
                        dtype=torch.float32):
    i = torch.arange(T + 1, device=device, dtype=dtype)
    t = i / T

    f  = torch.cos(((t + s) / (1 + s)) * torch.pi / 2)   # tensor ok (broadcasted s)
    f0 = math.cos((s / (1 + s)) * math.pi / 2)           # scalar -> use math.cos

    abar = (f / f0).clamp(min=1e-12) ** 2
    a    = (abar[1:] / abar[:-1]).clamp(1e-5, 0.999)
    beta = 1.0 - a
    return beta


def run_ffs_model():
    
    batch_size = 16
    #K = 4  # data augmentation factor
    n_epochs = 500
    lr = 1e-3
    truncate_training_data_to = 100
    n_inference_steps = 50
    perform_inference_every = 75
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #tfm = transforms.Compose([transforms.Resize((8,8), antialias=True), transforms.ToTensor()])
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    if truncate_training_data_to is not None:
        ds.data = ds.data[:truncate_training_data_to]
        ds.targets = ds.targets[:truncate_training_data_to]
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    C, H, W = next(iter(dl))[0].shape[1:]  # get image shape

    #model = models.SimpleResNetWithTimeInput(1, 32, 16, device=device)

    model = UNet2DModel(
        sample_size=28,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x, _ in dl:
            B = x.size(0)
            x1 = x.to(device)*2-1.0  # scale to [-1,1]
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device).reshape(B, 1, 1, 1)
            xt = (1-t) * x0 + t * x1
            v = x1 - x0
            v_pred = model(xt.float(), (1000*t).reshape(B)).sample   # predict v
            loss = criterion(v_pred, v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * B
        epoch_loss /= len(dl.dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, LossPerItem: {epoch_loss:.4f}")

        if (epoch + 1) % perform_inference_every == 0:
            model.eval()
            with torch.no_grad():
                n_gen = 16
                x_t = torch.randn((n_gen, C, H, W), device=device)  # start from pure noise
                for step in range(n_inference_steps):
                    t_scalar = float(step / n_inference_steps)
                    t_batch = torch.full((n_gen,), t_scalar, device=device, dtype=torch.float32)
                    v_pred = model(x_t.float(), (1000 * t_batch)).sample
                    x_t = x_t + v_pred / n_inference_steps
                x_t = (x_t + 1.0) / 2.0  # scale back to [0,1]
                x_train_sample = next(iter(dl))[0][:16].to(device)
                x_t = torch.cat([x_train_sample, x_t], dim=0)
                x_t.clamp_(0.0, 1.0)
                grid = vutils.make_grid(x_t.cpu(), nrow=8, normalize=True, pad_value=1.0)
                print("Grid shape:", grid.shape)
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis("off")
                plt.show()
            model.train()

def run_ffs_midi_model():
    
    batch_size = 16
    #K = 4  # data augmentation factor
    n_epochs = 500
    lr = 1e-3
    truncate_training_data_to = 100
    n_inference_steps = 50
    perform_inference_every = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #tfm = transforms.Compose([transforms.Resize((8,8), antialias=True), transforms.ToTensor()])
    tfm = transforms.ToTensor()

    midi_paths = list(Path("temp_data/maestro-v3.0.0").rglob("*.midi"))
    ds = mydatasets.MaestroMIDIPianoRollImageDataset(
        midi_filenames=[str(p) for p in midi_paths[:1]],
        sample_hz=30,
        window_width=100,
        n_samples=100,
        materialize_all=False
    )

    #if truncate_training_data_to is not None:
    #    ds.data = ds.data[:truncate_training_data_to]
    #    ds.targets = ds.targets[:truncate_training_data_to]
    
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    C, H, W = next(iter(dl))[0].shape[1:]  # get image shape

    model = models.SimpleResNetWithTimeInput(1, 32, 16, device=device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x, _ in dl:
            B = x.size(0)
            x1 = x.to(device)*2-1.0  # scale to [-1,1]
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device).reshape(B, 1, 1, 1)
            xt = (1-t) * x0 + t * x1
            v = x1 - x0
            v_pred = model(xt.float(), (1000*t).reshape(B)).sample   # predict v
            loss = criterion(v_pred, v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * B
        epoch_loss /= len(dl.dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, LossPerItem: {epoch_loss:.4f}")

        if (epoch + 1) % perform_inference_every == 0:
            model.eval()
            with torch.no_grad():
                n_gen = 16
                x_t = torch.randn((n_gen, C, H, W), device=device)  # start from pure noise
                for step in range(n_inference_steps):
                    t_scalar = float(step / n_inference_steps)
                    t_batch = torch.full((n_gen,), t_scalar, device=device, dtype=torch.float32)
                    v_pred = model(x_t.float(), (1000 * t_batch)).sample
                    x_t = x_t + v_pred / n_inference_steps
                x_t = (x_t + 1.0) / 2.0  # scale back to [0,1]
                x_train_sample = next(iter(dl))[0][:16].to(device)
                x_t = torch.cat([x_train_sample, x_t], dim=0)
                x_t.clamp_(0.0, 1.0)
                grid = vutils.make_grid(x_t.cpu(), nrow=8, normalize=True, pad_value=1.0)
                print("Grid shape:", grid.shape)
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis("off")
                plt.show()
            model.train()


def run_diffusion_model():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    C, H, W = next(iter(dl))[0].shape[1:]  # get image shape

    model = models.SimpleResNetWithTimeInput(1, 32, 16, device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)

    n_steps = 100   # Num diffusion steps
    beta = cosine_abar_to_beta(n_steps, device=device)  # (n_steps,)
    alpha = 1.0 - beta
    alphabar = torch.cumprod(alpha, dim=0)    # (n_steps,)

    n_epochs = 150
    gen_val_samples_every = 3
    K = 4

    for i in range(n_epochs):
        epoch_loss = 0.0
        for _batch, _ in dl:
            optimizer.zero_grad()
            batch = _batch.repeat_interleave(K, dim=0)  # augment data by factor K
            B = batch.size(0)
            batch = batch.to(device)
            t = torch.randint(0, n_steps, (B,)).long().to(device)   # sample time step t
            noise = torch.randn_like(batch).to(device)   # sample noise e ~ N(0,I)
            abar_t = alphabar[t]   # (B,)
            x_t = torch.sqrt(alphabar[t])[:, None, None, None] * batch + \
                  torch.sqrt(1 - alphabar[t])[:, None, None, None] * noise   # noisy image at time step t
            noise_pred = model(x_t.float(), t.float()/n_steps)   # predict noise
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dl.dataset)
        print(f"Epoch {i+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        if i % gen_val_samples_every == 0 and i > 0:
            # Generate samples
            model.eval()
            with torch.no_grad():
                n_gen = 16
                x_t = torch.randn((n_gen, C, H, W), device=device)  # start from pure noise
                for t in reversed(range(n_steps)):
                    t_batch = torch.full((n_gen,), t, device=device, dtype=torch.float32)
                    noise_pred = model(x_t.float(), t_batch.float()/n_steps)
                    abar_t = alphabar[t]
                    abar_t_prev = alphabar[t-1] if t > 0 else torch.tensor(1.0).to(device)
                    coef1 = 1 / torch.sqrt(torch.tensor(alpha[t]).to(device))
                    coef2 = (1 - alpha[t]) / torch.sqrt(1 - abar_t)
                    x_t = coef1 * (x_t - coef2 * noise_pred)
                    if t > 0:
                        noise = torch.randn_like(x_t).to(device)
                        sigma_t = torch.sqrt((1 - abar_t_prev) / (1 - abar_t) * beta[t])
                        x_t += sigma_t * noise
                # x_0 is the generated sample
                print("Generated samples at epoch", i+1)
                x_train_sample = next(iter(dl))[0][:16].to(device)
                x_t = torch.cat([x_train_sample, x_t], dim=0)
                x_t.clamp_(0.0, 1.0)
                grid = vutils.make_grid(x_t.cpu(), nrow=8, normalize=True, pad_value=1.0)
                print("Grid shape:", grid.shape)
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis("off")
                plt.show()
            model.train()



def run_autoencoder(dl: torch.utils.data.DataLoader):
    model = models.SimpleResNet(1, 16)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)

    n_epochs = 100

    for i in range(n_epochs):
        epoch_loss = 0.0
        for batch in dl:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dl.dataset)
        print(f"Epoch {i+1}/{n_epochs}, Loss: {epoch_loss:.4f}")



def main():

    #run_autoencoder(dl)
    #run_diffusion_model()
    #run_ffs_model()
    run_ffs_midi_model()



if __name__ == "__main__":
    main()