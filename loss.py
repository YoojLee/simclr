# NT-Xent Loss 구현 (contrastive loss의 한 종류)
import torch
import torch.nn as nn

LARGE_NUMBER = -1e9

class NTXentLoss(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.temp = temp

    def forward(self, sim, y):
        """
        - Args
            sim: a similarity matrix (2N X 2N)
        """
        # masking a diagonal of a similarity matrix.
        mask = torch.ones_like(sim)
        mask.fill_diagonal_(LARGE_NUMBER) # large number 채우기

        # apply a mask to a similarity matrix.
        sim *= mask

        loss = self.ce(sim/self.temp, y)

        return loss   