import torch.nn as nn
import torch
from typing import List, Callable
import math
from enum import Enum, auto



class ScanOrder(Enum):
    ROW_MAJOR = auto()
    COLUMN_MAJOR = auto()


def flatten_and_discretize(batch:torch.Tensor, 
                          order: ScanOrder):
    assert len(batch.shape)==4, f"This function assumes B,C,H,W, but the shape is {batch.shape}"
    B,C,H,W = batch.shape
    if order == ScanOrder.COLUMN_MAJOR:
        batch = batch.permute(0,1,3,2)
        batch = batch.flatten(1)
        batch = torch.bucketize(batch, [0.05])
        assert torch.all((0 <= batch) & (batch <= 1)).item()
        return batch
    elif order == ScanOrder.ROW_MAJOR:
        batch = batch.flatten(1)
        batch = torch.bucketize(batch, [0.05])
        assert torch.all((0 <= batch) & (batch <= 1)).item()
        return batch
    else:
        assert False, f"Invalid scan order: {order}"



