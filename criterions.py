import torch.nn as nn
import torch
from torch.nn.modules.loss import BCELoss

class BPR(nn.Module):
    def __init__(self):
        super(BPR, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='sum')
    def forward(self, score_l_items, score_r_items, labels):
        diffs = score_l_items - score_r_items
        probs = torch.sigmoid(diffs) # If score of seen item > score of unseen item, the probability is greater than 0.5
        neg_llh = self.bce_loss(probs, labels)
        return neg_llh

