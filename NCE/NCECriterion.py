import torch
import torch.nn as nn
from NCE.NCEAverage import NCEAverage

eps = 1e-7

class NCECriterion(nn.Module):
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.size(0)
        K = x.size(1) - 1

        # noise distribution
        Pn = 1. / self.n_data

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(K * Pn + eps)).log()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, K)
        log_D0 = torch.div(P_neg.clone().fill_(K * Pn), P_neg.add(K * Pn + eps)).log()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


if __name__ == '__main__':
    average = NCEAverage(128, 9000, 4, 0.07, 3).cuda()
    criterion = NCECriterion(9000).cuda()
    # print(average.pos_indices)
    # print(average.neg_indices)
    dummy_embeddings = torch.randn(12, 128).cuda()
    idxs = torch.arange(3).cuda()

    outs, probs, new_data_memory = average(dummy_embeddings, idxs)
    print(outs[:, 0].mean())
    print(probs)
    loss = criterion(outs).item()
    print(loss)