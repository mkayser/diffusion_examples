from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from pathlib import Path
import mido
import logging
from utils import NullLog


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



def dump_midi(path):
    for msg in mido.MidiFile(path):
        msg_vars = vars(msg)
        if msg_vars['type'] == "note_on":
            assert all(key in msg_vars for key in ['channel', 'note', 'velocity', 'time'])
            print(f"Valid note: {msg}")
        else:
            print(f"Removed: {msg}")



@dataclass
class MidiNotePress:
    time: float  # in seconds
    duration: float  # in seconds
    note: int    # MIDI note number
    velocity: int  # 0-127

def extract_note_presses_from_midi(path: str,
                                   logger = NullLog()) -> List['MidiNotePress']:
    midi = mido.MidiFile(path)
    note_presses: List[MidiNotePress] = []
    current_time = 0.0

    unresolved_note_presses = {}  # key: note number, value: (start_time, velocity)
    n_messages = 0

    for msg in midi:
        n_messages += 1
        current_time += msg.time
        if msg.type == "note_on":
            if msg.velocity > 0:
                # Note on
                unresolved_note_presses[msg.note] = (current_time, msg.velocity)
            else:
                # Note off (velocity 0)
                if msg.note in unresolved_note_presses:
                    start_time, velocity = unresolved_note_presses.pop(msg.note)
                    duration = current_time - start_time
                    note_presses.append(MidiNotePress(time=start_time, duration=duration, note=msg.note, velocity=velocity))
                else:
                    logger.warn(f"Warning: Note off for note {msg.note} at time {current_time} without matching note on.")
        elif msg.type == "note_off":
            logger.warn(f"Warning: Note off message encountered for note {msg.note}. This implementation expects note_on with velocity 0 for note offs.")

    for note, (start_time, velocity) in unresolved_note_presses.items():
        logger.warn(f"Warning: Note {note} started at {start_time} has no matching note off.")

    logger.info(f"Extracted {len(note_presses)} note presses from {n_messages} total messages in file {path}")
    return note_presses

def note_presses_to_piano_roll_image(note_presses: List['MidiNotePress'],
                                    sample_hz: int) -> torch.Tensor:
    t_final = max(note.time + note.duration for note in note_presses)
    n_time_steps = int(t_final * sample_hz) + 1
    H = 88  # piano key range from A0 (21) to C8 (108)
    W = n_time_steps
    piano_roll = torch.zeros((1, H, W), dtype=torch.float32)
    for note in note_presses:
        start_idx = int(note.time * sample_hz)
        end_idx = int((note.time + note.duration) * sample_hz)
        piano_key = note.note - 21  # map MIDI note to piano key index
        if 0 <= piano_key < H:
            piano_roll[0, piano_key, start_idx:end_idx] = note.velocity / 127.0  # normalize velocity
    return piano_roll


class MaestroMIDIPianoRollImageDataset(Dataset):
    def __init__(self, 
                 midi_filenames: List[str],
                 sample_hz: int, 
                 window_width: int,
                 n_samples: int,
                 materialize_all: bool = False,
                 rng_seed: int = 42):
        self.sample_hz:int = sample_hz
        self.C = 1
        self.H:int = 88
        self.W:int = window_width
        self.n = n_samples
        self.sr = sample_hz
        self.materialize_all = materialize_all
        self.backing_images:List[torch.Tensor] = list(self._load_backing_images(midi_filenames))
        self.rng = torch.Generator().manual_seed(rng_seed)

        self.precomputed_windows = list(self._load_precomputed_windows())

        if self.materialize_all:
            self.data = self._load_windows(self.n)

    def _load_precomputed_windows(self):
        assert self.backing_images is not None
        assert len(self.backing_images) > 0
        image_lengths = [img.shape[2] for img in self.backing_images]
        valid_start_indices = [image_length - self.W for image_length in image_lengths]
        n_valid_indices = sum(valid_start_indices)
        assert n_valid_indices > 0, "No valid windows available in backing images."
        chosen_indices = torch.randint(0, n_valid_indices, (self.n,), generator=self.rng)
        for idx in chosen_indices:
            # Find which backing image this index falls into
            cumulative = 0
            for img_idx, valid_length in enumerate(valid_start_indices):
                if idx < cumulative + valid_length:
                    start_idx = idx - cumulative
                    #yield self.backing_images[img_idx][:, :, start_idx:start_idx + self.W]
                    yield (img_idx, start_idx)
                    break
                cumulative += valid_length

    def _load_backing_images(self, midi_filenames:str):
        for filename in midi_filenames:
            note_presses = extract_note_presses_from_midi(filename)
            piano_roll = note_presses_to_piano_roll_image(note_presses, self.sample_hz)
            print(f"Piano roll shape for {filename}: {piano_roll.shape}")
            yield piano_roll

    def _load_windows(self, n:int):
        assert False, "Not implemented yet"

    def __len__(self): 
        return self.n

    def __getitem__(self, i):
        if self.materialize_all:
            assert False, "Not implemented yet"
        else:
            img_idx, start_idx = self.precomputed_windows[i]
            img = self.backing_images[img_idx][:, :, start_idx:start_idx + self.W]
            return img
        

if __name__ == "__main__":

    midi_paths = list(Path("temp_data/maestro-v3.0.0").rglob("*.midi"))
    ds = MaestroMIDIPianoRollImageDataset(
        midi_filenames=[str(p) for p in midi_paths[:10]],
        sample_hz=30,
        window_width=100,
        n_samples=1000,
        materialize_all=False
    )

    dl = DataLoader(ds, batch_size=16, shuffle=True)
    for batch in dl:
        print(batch.shape)
        grid = vutils.make_grid(batch, nrow=4, normalize=True, pad_value=1.0)
        print("Grid shape:", grid.shape)
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
        exit(0)

    exit(0)
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