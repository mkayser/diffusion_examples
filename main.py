import torch.nn as nn
import torch
import datasets
import models
import numpy as np



def run_diffusion_model(dl: torch.utils.data.DataLoader):
    model = models.SimpleResNet(1, 16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_steps = 1000   # Num diffusion steps
    beta = np.linspace(0.0001, 0.02, n_steps)
    alpha = 1.0 - beta
    alphabar = np.cumprod(alpha)



    for i in range(n_epochs):
        epoch_loss = 0.0
        # sample: e ~ N(0,I)
        # sample: t ~ distr(T)
        # training pair: (x+v[i]e,i), 


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
    ds = datasets.PerpendicularLinesDataset(4096, (1, img_hw, img_hw), 128)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    run_autoencoder(dl)



if __name__ == "__main__":
    main()