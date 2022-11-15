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
            y: labels
        """
        sim.fill_diagonal_(LARGE_NUMBER)

        loss = self.ce(sim/self.temp, y)

        return loss   