import torch.nn as nn
import torch
import datasets
import models
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import math


def run_diffusion_model(dl: torch.utils.data.DataLoader):
    print(dir(models))
    model = models.SimpleResNetWithTimeInput(1, 32, 16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_steps = 10000   # Num diffusion steps
    beta = np.linspace(0.0001, 0.02, n_steps)
    alpha = 1.0 - beta
    alphabar = torch.from_numpy(np.cumprod(alpha))  # (n_steps,)

    n_epochs = 1500
    gen_val_samples_every = 200

    for i in range(n_epochs):
        epoch_loss = 0.0
        for batch in dl:
            optimizer.zero_grad()
            B = batch.size(0)
            t = torch.randint(0, n_steps, (B,)).long()   # sample time step t
            #print("t shape:", t.shape)
            #print ("alphabar shape:", alphabar.shape)
            #print ("batch shape:", batch.shape)
            noise = torch.randn_like(batch)   # sample noise e ~ N(0,I)
            abar_t = alphabar[t]   # (B,)
            #print ("abar_t shape:", abar_t.shape)
            x_t = torch.sqrt(alphabar[t])[:, None, None, None] * batch + \
                  torch.sqrt(1 - alphabar[t])[:, None, None, None] * noise   # noisy image at time step t
            #print("x_t shape:", x_t.shape, "t shape:", t.shape)
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
                x_t = torch.randn((n_gen, 1, 16, 16))  # start from pure noise
                for t in reversed(range(n_steps)):
                    t_batch = torch.full((n_gen,), t, dtype=torch.float32)
                    noise_pred = model(x_t.float(), t_batch.float()/n_steps)
                    abar_t = alphabar[t]
                    abar_t_prev = alphabar[t-1] if t > 0 else torch.tensor(1.0)
                    coef1 = 1 / torch.sqrt(torch.tensor(alpha[t]))
                    coef2 = (1 - alpha[t]) / torch.sqrt(1 - abar_t)
                    x_t = coef1 * (x_t - coef2 * noise_pred)
                    if t > 0:
                        noise = torch.randn_like(x_t)
                        sigma_t = torch.sqrt((1 - abar_t_prev) / (1 - abar_t) * beta[t])
                        x_t += sigma_t * noise
                # x_0 is the generated sample
                print("Generated samples at epoch", i+1)

                grid = vutils.make_grid(x_t, nrow=4, normalize=True, pad_value=1.0)
                print("Grid shape:", grid.shape)
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis("off")
                plt.show()
            model.train()



def run_autoencoder(dl: torch.utils.data.DataLoader):
    model = models.SimpleResNet(1, 16)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
    img_hw = 16
    num_lines = 20
    ds = datasets.PerpendicularLinesDataset(4096, (1, img_hw, img_hw), num_lines)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    #run_autoencoder(dl)
    run_diffusion_model(dl)



if __name__ == "__main__":
    main()