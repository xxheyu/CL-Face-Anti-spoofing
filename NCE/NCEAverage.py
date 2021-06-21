import torch
import torch.nn as nn
import math
import functools
import operator
import pdb
import numpy as np
import torch.nn.functional as F

from .alias_multinomial import AliasMethod


class NCEAverage(nn.Module):
    def __init__(self, inputSize, nLen, sample_per_class, T, num_classes, Z_momentum=0.9, memory_momentum=0.5):
        super(NCEAverage, self).__init__()
        print("[NCE]: Initialization...")
        self.nLen = nLen
        self.embed_dim = inputSize
        self.sample_per_class = sample_per_class

        self.unigrams = torch.ones(self.nLen)
        #self.multinomial = AliasMethod(self.unigrams)
        #self.multinomial.cuda()
        self.K = 4096


        # self.number_combinations = math.factorial(self.sample_per_class) // (math.factorial(self.sample_per_class - 2) * 2)
        # print("[NCE]    Got {} combinations in full graph version".format(self.number_combinations))

        # Z, T, Z_momentum
        self.register_buffer('params', torch.tensor([-1, T, Z_momentum, memory_momentum, -1]))
        print("[NCE]: params Z {}, T {}, Z_momentum {}, memory_momentum {}".format(-1, T, Z_momentum, memory_momentum))
        # memory bank for test (kNN)
        # stdv = 1. / math.sqrt(self.embed_dim / 3)
        # self.register_buffer('memory', torch.rand(self.nLen, self.embed_dim).mul_(2 * stdv).add_(-stdv))
        print("[NCE]: memory bank, {} {}-dim random unit vectors".format(self.nLen, self.embed_dim))

        self.prepare_indices(num_classes, (num_classes)*self.sample_per_class)

    def remove_self(self, V_out):
        # ======== remove diagonal elements of V_out ========== #
        assert V_out.size(0) == V_out.size(1)

        return V_out[~torch.eye(V_out.shape[0], dtype=torch.bool)].view(V_out.shape[0], -1)

    def reorder_out(self, V_out):
        # ======== move positive pair to the first column ======= #
        assert V_out.size(0) == V_out.size(1)
        positive_pairs = torch.diag(V_out).unsqueeze(1)
        negative_pairs = self.remove_self(V_out)

        return torch.cat([positive_pairs, negative_pairs], dim=1)

    def prepare_indices(self, batch_size, bs):
        '''this func prepare pos_idx and neg_idx for each sample clip
        each sample clip has sample_per_class-1 positive pairs and (batch_size-1)*sample_per_class negative pairs
        this func is done once in initialization, and needs to be invoked again for the last batch
        since the batch_size will be different
        '''
        self.batch_size = batch_size
        self.bs = bs
        self.indices = torch.arange(self.bs).view(self.sample_per_class, self.batch_size).t().cuda()  # this is for computing memory update
        self.pos_indices = torch.zeros(self.bs, self.sample_per_class-1).cuda()
        self.neg_indices = torch.zeros(self.bs, (self.batch_size-1) * self.sample_per_class).cuda()
        indices_temp = []
        for i in range(self.batch_size):
            indices_temp.append([i * self.sample_per_class + ii for ii in range(self.sample_per_class)])
            # indices_temp.append([i + ii * self.batch_size for ii in range(self.sample_per_class)])
        for i in range(self.bs):
            pos_temp = indices_temp[i // self.sample_per_class].copy()
            pos_temp.remove(i)
            neg_temp = indices_temp.copy()
            neg_temp.pop(i // self.sample_per_class)
            self.pos_indices[i, :] = torch.tensor(pos_temp).cuda()  # (sample_per_class-1)
            self.neg_indices[i, :] = torch.tensor(functools.reduce(operator.iconcat, neg_temp, [])).cuda()  # ((batch_size-1)*sample_per_class, )

        self.pos_indices = self.pos_indices.long()
        self.neg_indices = self.neg_indices.long()


    def compute_data_prob(self):
        positives = self.embeddings[self.pos_indices]  # (bs, sample_per_class-1, embed_dim)
        prods = self.embeddings.unsqueeze(1) * positives  # (bs, 1, embed_dim) * (bs, sample_per_class-1, embed_dim)
        logits = torch.sum(prods, dim=-1)  # outcome of innerproduct between features -- (bs, sample_per_class-1)
        logits = torch.mean(logits, dim=-1, keepdim=True)  # (bs, 1)
        # logits = torch.min(logits, 1, keepdim=True)[0]  # HARD POSITIVE MINING

        return logits

    def compute_noise_prob(self):
        # neg_size = (batch_size-1) * sample_per_class
        negatives = self.embeddings[self.neg_indices]  # (bs, neg_size, embed_dim)
        prods = self.embeddings.unsqueeze(1) * negatives  # (bs, 1, embed_dim) * (bs, neg_size, embed_dim)
        logits = torch.sum(prods, dim=-1)  # outcome of innerproduct between features -- (bs, neg_size)

        return logits


    def nce_core(self, pos_logits, neg_logits):
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        outs = torch.exp(logits / self.params[1].item())
        Z = self.params[0].item()
        if Z < 0:
            # initialize it with mean of first batch
            self.params[0] = outs.mean() * self.nLen
            Z = self.params[0].clone().detach().item()
            print('normalization constant Z is set to {:.1f}'.format(Z))
        else:
            Z_new = outs.mean() * self.nLen
            self.params[0] = (1 - self.params[2]) * Z_new + self.params[2] * self.params[0]
            Z = self.params[0].clone().detach().item()

        outs = torch.div(outs, Z).contiguous()
        probs = self.extract_probs(outs)

        return outs, probs

    def extract_probs(self, out):
        probs = out / torch.sum(out, dim=1, keepdim=True)

        return probs[:, 0].mean()

    def updated_new_data_memory(self, idxs):
        data_memory = torch.index_select(self.memory, 0, idxs)
        # pdb.set_trace()
        new_data_memory = self.embeddings
        new_data_memory = data_memory * self.params[3] + (1 - self.params[3]) * new_data_memory ### LOOK HERE
        # new_data_memory = l2_normalize(new_data_memory) ### LOOK HERE

        idxs = idxs.unsqueeze(1).repeat(1, self.embed_dim)
        self.memory.scatter_(0, idxs, new_data_memory)

    def recompute_memory(self, idxs):
        new_data_memory = torch.mean(self.embeddings[self.indices], dim=1)
        new_data_memory = l2_normalize(new_data_memory)
        idxs = idxs.unsqueeze(1).repeat(1, self.embed_dim)
        self.memory.scatter_(0, idxs, new_data_memory)


    def forward(self, x, i):
        '''Only use intra batch noise samples, memory bank is only for test(kNN)
        x: embeddings (bs, embed_dim) where bs = args.batch_size * sample_per_class

        '''
        bs = x.size(0)
        batch_size = bs // self.sample_per_class
        self.embeddings = x
        # treat the last batch specially
        if i == self.nLen - 1:
            # pdb.set_trace()
            self.prepare_indices(batch_size, bs)
        # self.prepare_indices()

        pos_logits = self.compute_data_prob()
        neg_logits = self.compute_noise_prob()
        outs, probs = self.nce_core(pos_logits, neg_logits)

        # with torch.no_grad():
        #     self.updated_new_data_memory(idxs)
        #     # self.recompute_memory(idxs)

        return outs, probs

if __name__ == '__main__':
    average = NCEAverage(128, 9000, 4, 0.07, 3).cuda()
    print(average.pos_indices)
    print(average.neg_indices)
    dummy_embeddings = torch.randn(12, 128).cuda()
    dummy_embeddings = l2_normalize(dummy_embeddings)
    idxs = torch.arange(3).cuda()
    outs, probs, new_data_memory = average(dummy_embeddings, idxs)
    print(outs)
    print(outs.shape)
    # pdb.set_trace()
