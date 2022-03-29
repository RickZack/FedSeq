import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss


class SoftTarget(nn.Module):
    '''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss


class KlDistillationLoss(nn.Module):
    def __init__(self, T: float, alpha: float):
        super(KlDistillationLoss, self).__init__()
        self.crEntr = CrossEntropyLoss()
        self.dist = SoftTarget(T)
        self.alpha = alpha

    def forward(self, out_s, out_t, target):
        loss = self.crEntr(out_s, target) + self.alpha * self.dist(out_s, out_t)
        return loss
