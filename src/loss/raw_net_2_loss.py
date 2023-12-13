import torch
from torch import nn
import torch.nn.functional as F

class RawNet2Loss(nn.Module):
    def __init__(self, bonafide_weight=9.0, spoof_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor([spoof_weight, bonafide_weight]))

    def forward(self, prediction, is_bonafide, **batch):
        return self.loss(prediction, is_bonafide)
