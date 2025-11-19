import datasets
import torch
from enum import Enum, auto
from typing import Optional
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
import datasets as mydatasets


class Discretization(Enum):
    ON_OR_OFF = auto()


class PianoRollImageToIntSeqConverter:
    def __init__(self, 
                 discretization: Discretization,
                 image_shape: tuple[int,int],
                 max_seq_len: int = 2000):
        self.discretization = discretization
        self.image_shape = image_shape  # (H,W)
        assert len(image_shape) == 2
        H, W = image_shape
        self.pad_token_id = 0
        self.bos_token_id = H+1
        self.eos_token_id = H+2
        self.delim_token_id = H+3
        self.max_seq_len = max_seq_len

    def convert(self, imgs: torch.Tensor) -> torch.Tensor:
        assert len(imgs.shape) == 4, f"Expected imgs to have shape B,C,H,W but got {imgs.shape}"
        assert imgs.shape[1] == 1, f"Expected imgs to have C=1 but got C={imgs.shape[1]}"
        assert imgs.shape[2:] == self.image_shape, f"Expected imgs to have shape H,W={self.image_shape} but got H,W={imgs.shape[2:]}"
        imgs = imgs.squeeze(1).contiguous()  # B,H,W
        
        B, H, W = imgs.shape

        DELIM, BOS, EOS = self.delim_token_id, self.bos_token_id, self.eos_token_id

        if self.discretization == Discretization.ON_OR_OFF:
            imgs = torch.bucketize(imgs, torch.tensor([0.05], device=imgs.device))

            lens = []
            seqs = []
            for b in range(B):
                img = imgs[b]  # H,W
                seq = [torch.tensor([BOS])]
                _len = 1
                for c in range(W):
                    col = img[:,c]
                    nz = torch.nonzero(col).flatten()  #1d tensor of nonzero row positions
                    nz = (H - nz) + 1    # 1d tensor of row positions, 1-indexed, starting from bottom row
                    nz = nz.flip(0)  # ascending order
                    seq.append(nz)
                    seq.append(torch.tensor([DELIM]))
                    _len += nz.numel() + 1
                seq.append(torch.tensor([EOS]))
                _len += 1
                seqs.append(seq)
                lens.append(_len)
            maxlen = max(lens)
            L = min(maxlen, self.max_seq_len) # Truncated max length
            for s,l in zip(seqs, lens):
                if l < L:
                    pad = torch.zeros(L-l, dtype=imgs.dtype)
                    s.append(pad)
            tensors = [torch.concat(s)[:L] for s in seqs]
            output = torch.stack(tensors)
            return output
        else:
            assert False, f"Invalid discretization: {self.discretization}"



class VanillaTransformerModel(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 d_hidden: int,
                 vocab_size: int,
                 max_seq_len: int,
                 pad_token_id: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 delim_token_id: int):
        super(VanillaTransformerModel, self).__init__()
        self.V = vocab_size
        self.H = d_hidden
        self.input_emb = nn.Embedding(self.V, self.H)
        self.pos_emb = nn.Embedding(max_seq_len, self.H)
        encoder_layer = nn.TransformerEncoderLayer(self.H, 8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_emb = nn.Linear(self.H, self.V, bias=False)
        # The transformer does not use these values directly, but they are stored here for reference.
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.delim_token_id = delim_token_id

    def _causal_mask(self, n:int, device:str|torch.device):
        mask = torch.full((n,n), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor):
        assert len(input_ids.shape)==2
        B,T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device)
        input_emb = self.input_emb(input_ids)
        pos_emb = self.pos_emb(positions).unsqueeze(0)
        x = input_emb + pos_emb
        mask = self._causal_mask(T, device)
        hidden = self.transformer(x, mask = mask)
        logits = self.output_emb(hidden)
        assert logits.shape == (B,T,self.V)
        return logits


def transformer_train_step(model: VanillaTransformerModel, 
                           batch: torch.Tensor,
                           criterion: nn.Module,
                           optimizer: torch.optim.Optimizer):
    assert len(batch.shape) == 2
    B,T = batch.shape

    logits = model(batch)
    assert len(logits.shape) == 3
    assert logits.shape[0] == B and logits.shape[1] == T
    V = logits.shape[2]

    model_outputs = logits[:,:-1,:].reshape(-1,V)
    ref_output_ids = batch[:,1:].flatten()

    loss = criterion(model_outputs, ref_output_ids)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def transformer_train():

    batch_size = 16
    n_epochs = 100
    lr = 1e-4
    truncate_training_data_to = 100
    n_inference_steps = 50
    perform_inference_every = 20
    max_seq_len = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(42)

    midi_paths = list(Path("temp_data/maestro-v3.0.0").rglob("*.midi"))
    ds = mydatasets.MaestroMIDIPianoRollImageDataset(
        midi_filenames=[str(p) for p in midi_paths[:1]],
        sample_hz=30,
        window_width=100,
        n_samples=truncate_training_data_to,
        materialize_all=False
    )

    converter = PianoRollImageToIntSeqConverter(
        discretization=Discretization.ON_OR_OFF,
        image_shape=(88,100),
        max_seq_len=max_seq_len
    )

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)
    C, H, W = next(iter(dl))[0].shape  # get image shape
    print("C,H,W:", C, H, W)

    model = VanillaTransformerModel(
        num_layers=4,
        d_hidden=256,
        vocab_size=H+4,  # note values + BOS + EOS + DELIM + PAD
        max_seq_len=max_seq_len,
        pad_token_id=0,
        bos_token_id=H+1,
        eos_token_id=H+2,
        delim_token_id=H+3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for imgs in dl:
            x = converter.convert(imgs)  # B,L
            x = x.to(device)
            loss = transformer_train_step(model, x, criterion, optimizer)
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(dl.dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, LossPerItem: {epoch_loss:.4f}")

        if (epoch + 1) % perform_inference_every == 0:
            model.eval()
            with torch.no_grad():
                pass
            model.train()


        

if __name__ == "__main__":
    transformer_train()
    # print("Testing convert_pianoroll_images_to_int_seqs...")
    # midi_paths = list(Path("temp_data/maestro-v3.0.0").rglob("*.midi"))

    # ds = datasets.MaestroMIDIPianoRollImageDataset(
    #     midi_filenames=midi_paths[:1],
    #     sample_hz=10,
    #     window_width=5,
    #     n_samples=1,
    #     materialize_all=False
    # )
    # img = torch.unsqueeze(ds[0],0)  # 1,C,H,W
    # assert img.shape == (1,1,88,5), f"Unexpected image shape: {img.shape}"
    # print("Input image shape:", img.shape)
    # print("Input image:", img)
    # int_seq = convert_pianoroll_images_to_int_seqs(img, Discretization.ON_OR_OFF, delimiter=None)
    # print("Output int seq shape:", int_seq.shape)
    # print("Output int seq:", int_seq)