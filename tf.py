import datasets
import torch
from enum import Enum, auto
from typing import Optional
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datasets as mydatasets
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import math
import time



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

    def convert_back(self, seqs: torch.Tensor) -> torch.Tensor:
        assert len(seqs.shape) == 2, f"Expected seqs to have shape B,L but got {seqs.shape}"
        B, L = seqs.shape
        H, W = self.image_shape
        output_imgs = torch.zeros((B,1,H,W), dtype=torch.float32, device=seqs.device)
        PAD, DELIM, BOS, EOS = self.pad_token_id, self.delim_token_id, self.bos_token_id, self.eos_token_id

        if self.discretization == Discretization.ON_OR_OFF:
            for b in range(B):
                seq = seqs[b]  # L
                img = torch.zeros((H,W), dtype=torch.float32, device=seqs.device)
                w = 0
                i = 0
                while i < L:
                    token = seq[i].item()
                    if token == BOS:
                        i += 1
                    elif token == PAD:
                        break
                    elif token == EOS:
                        break
                    elif token == DELIM:
                        w += 1
                        i += 1
                    else:
                        row = H - (token - 1)
                        if 0 <= row < H and 0 <= w < W:
                            img[row, w] = 1.0
                        i += 1
                output_imgs[b,0,:,:] = img
        else:
            assert False, f"Invalid discretization: {self.discretization}"
        return output_imgs




class PositionalEmbeddingType(Enum):
    LEARNED_RAW_INDEX = auto()


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_pos: int = 10000, base: int = 10000):
        super().__init__()
        # 1. Handle Odd Dimensions
        if dim % 2 != 0:
            raise ValueError("Embedding dimension must be even for sinusoidal encoding")
            
        half = dim // 2
        # 2. Compute frequencies once (The logic from your snippet)
        freqs = torch.exp(-math.log(base) * torch.arange(half) / half)
        
        # 3. Create the grid (0 to max_pos)
        # We use .arange instead of passing 't' in
        t = torch.arange(max_pos) 
        args = t[:, None] * freqs[None] # (Seq_Len, half)
        
        # 4. Create the embedding
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # (Seq_Len, dim)
        
        # 5. Register as buffer so it saves with model but doesn't update via gradient
        # Unsqueeze to (1, Seq_Len, Dim) to broadcast over Batch
        self.register_buffer('pe', emb) 

    def forward(self, x: torch.Tensor):
        # x shape: (Batch, Seq_Len)
        assert x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64), f"Positional embedding requires integer tensor input"
        assert len(x.shape) == 2, f"Positional embedding expected shape of length 2 (B, L) but got shape = {x.shape}"
        B,L = x.shape
        minpos = 0
        maxpos = self.pe.shape[1]
        assert torch.all((x >= minpos) & (x <= maxpos)), f"Positional embedding elements must all be between {minpos} and {maxpos}"
        return F.embedding(x, self.pe)


class VanillaTransformerModel(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 d_hidden: int,
                 vocab_size: int,
                 max_pos: int,
                 pad_token_id: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 delim_token_id: int
                 ):
        super(VanillaTransformerModel, self).__init__()
        self.V = vocab_size
        self.H = d_hidden
        self.input_emb = nn.Embedding(self.V, self.H)
        self.pos_emb = SinusoidalPositionalEmbedding(self.H, max_pos)
        encoder_layer = nn.TransformerEncoderLayer(self.H, 8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_emb = nn.Linear(self.H, self.V, bias=False)
        # The transformer does not use these values directly, but they are stored here for reference.
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.delim_token_id = delim_token_id

    def _pos_emb(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    def _causal_mask(self, n:int, device:str|torch.device):
        mask = torch.full((n,n), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor):
        assert len(input_ids.shape)==2
        B,T = input_ids.shape
        device = input_ids.device

        # compute positions: for each b,t, pos[b,t] = # delim's that have been observed to the left
        is_delim = (input_ids == self.delim_token_id)
        positions = torch.cumsum(is_delim, dim=1)

        input_emb = self.input_emb(input_ids)
        pos_emb = self.pos_emb(positions)
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


def transformer_inference(model: VanillaTransformerModel,
                          num_seqs: int,
                          max_seq_len: int,
                          temperature: float,
                          device: str|torch.device,
                          generator=torch.Generator):
    model.eval()
    generated_seqs = torch.full((num_seqs,1), model.bos_token_id, dtype=torch.long, device=device)  # B,1
    seq_ended = torch.zeros((num_seqs,), dtype=torch.bool, device=device)  # B,
    with torch.no_grad():
        for _ in range(max_seq_len-1):
            logits = model(generated_seqs)  # B,T,V
            next_token_logits = logits[:,-1,:]  # B,V
            probs = (next_token_logits / temperature).softmax(dim=-1)
            next_tokens = torch.multinomial(probs, 1, generator=generator)
            #next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # B,1
            generated_seqs = torch.cat([generated_seqs, next_tokens], dim=-1)  # B,T+1
            #print("Generated seqs shape:", generated_seqs.shape)
            seq_ended |= (next_tokens.squeeze(-1) == model.eos_token_id)
            seq_ended |= (next_tokens.squeeze(-1) == model.pad_token_id)
            if seq_ended.all() or generated_seqs.shape[1] >= max_seq_len:
                break

    model.train()
    return generated_seqs


def transformer_train():

    max_midi_files = 10
    batch_size = 16
    n_epochs = 500
    lr = 1e-4
    truncate_training_data_to = 5000
    perform_inference_every = 10
    max_seq_len = 500
    max_seq_len_inference = 500
    temperature = 1.0
    n_samples_to_generate = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(42)
    inference_generator = torch.Generator(device=device).manual_seed(42)

    midi_paths = list(Path("temp_data/maestro-v3.0.0").rglob("*.midi"))
    ds = mydatasets.MaestroMIDIPianoRollImageDataset(
        midi_filenames=[str(p) for p in midi_paths[:max_midi_files]],
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
        max_pos=max_seq_len,
        pad_token_id=0,
        bos_token_id=H+1,
        eos_token_id=H+2,
        delim_token_id=H+3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(n_epochs):
        t0 = time.time()
        epoch_loss = 0.0
        for imgs in dl:
            x = converter.convert(imgs)  # B,L
            x = x.to(device)
            loss = transformer_train_step(model, x, criterion, optimizer)
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(dl.dataset)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{n_epochs}, LossPerItem: {epoch_loss:.4f},  Time: {dt:.1f}s")

        if (epoch + 1) % perform_inference_every == 0:
            model.eval()
            with torch.no_grad():
                seq_ids = transformer_inference(model, 
                                                num_seqs=n_samples_to_generate, 
                                                max_seq_len=max_seq_len_inference,
                                                temperature=temperature, 
                                                device=device,
                                                generator=inference_generator)
                gen_imgs = converter.convert_back(seq_ids.cpu())  # B,C,H,W
                print(f"Generated {n_samples_to_generate} samples at epoch {epoch+1}, imgs shape: {gen_imgs.shape}")
                training_imgs = next(iter(dl))
                print(f"Training imgs shape: {training_imgs.shape}")
                all_imgs = torch.cat([training_imgs, gen_imgs], dim=0)

                grid = vutils.make_grid(all_imgs.cpu(), nrow=8, normalize=True, pad_value=1.0)
                print("Grid shape:", grid.shape)
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis("off")
                plt.show()

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