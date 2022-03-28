import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss


class LdisCl(nn.Module):
    """
    A distillation-based approach integrating continual learning
    and federated learning for pervasive services https://arxiv.org/pdf/2109.04197.pdf
    """

    def __init__(self, T: float):
        super(LdisCl, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = torch.mean(-F.softmax(out_t / self.T, dim=1) * F.log_softmax(out_s / self.T, dim=1))
        return loss


class FLwFLoss(nn.Module):
    def __init__(self, T: float, alpha: float):
        super(FLwFLoss, self).__init__()
        self.cross_entropy = CrossEntropyLoss()
        self.distillation = LdisCl(T)
        self.alpha = alpha

    def forward(self, out_s, out_t, target):
        loss = self.cross_entropy(out_s, target) + self.alpha * self.distillation(out_s, out_t)
        return loss
