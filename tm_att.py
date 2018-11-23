import torch
import torch.autograd as ag
import torch.nn as nn
import torch.optim as optim
import pdb
import torch.utils.data as data
import torch.nn.functional as F
import sklearn.metrics as metrics
import argparse
import math
from cnn_att import CNN_ATT

# from dataset import Dataset, Temporal_Data
# from dataset import collate_fn, collate_fn_temporal_for_pcnn, collate_fn_temporal, collate_fn1
# import numpy as np
# from character_process import n_letters
# from utils import pad_sequence
# torch.cuda.manual_seed(1)
# torch.manual_seed(1)

class TM_ATT(CNN_ATT):
    def __init__(self, settings):
        super(TM_ATT, self).__init__(settings)
        self.M_embed = nn.Parameter(torch.zeros(self.n_rel, self.feature_size), requires_grad=True)
        self.M_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)
        con = math.sqrt(6.0 / (self.n_rel + self.feature_size))
        nn.init.uniform_(self.M_embed, a=-con, b=con)
        nn.init.uniform_(self.M_bias, a=-con, b=con)

    def forward(self, input):
        bz = len(input)
        bags = [item['bag'] for item in input]
        labels = [item['label'] for item in input]
        features = self._create_sentence_embedding(bags, labels)
        bz = len(labels)
        if self.training:
            s = features[torch.arange(0, bz).long().cuda(), labels]
            # score is the same, but normalize over different set!
            scores = torch.matmul(s, self.r_embed.t()) + self.r_bias
            pred = self.pred_sm(scores)
        else:
            s = features
            scores = torch.matmul(s, self.r_embed.t()) + self.r_bias
            scores = scores.max(-1)[0]
            pred = self.pred_sm(scores)

        # transmision
        tmp = torch.matmul(features, self.M_embed.t())
        tmp += self.M_bias
        T = self.atten_sm(tmp.view(-1, self.n_rel)).view(-1, self.n_rel, self.n_rel)

        # regularization
        reg_loss = 0
        # here the reg loss should be ? n_rel - reg_los??
        for i in range(T.size(0)):
            reg_loss += torch.trace(T[i])
        # reg_loss = self.n_rel - reg_loss

        # for connecting with nll loss
        epsilon = 1e-12
        # in case for nan
        out = torch.log(torch.bmm(self.atten_sm(scores.unsqueeze(1)), T) + epsilon).squeeze().view(bz, -1)

        if self.training:
            return pred, out, reg_loss
        else:
            return pred

    def _create_sentence_embedding(self, bags, labels):
        batch_features = []
        for ix, bag in enumerate(bags):
            # pdb.set_trace()
            label = labels[ix]
            features = []
            for item in bag:
                w2v = self.w2v(item.t()[0])
                # this may need some modification for further use.
                pos1 = self.pos1_embed(item[:, 1])
                pos2 = self.pos2_embed(item[:, 2])
                feature = torch.cat([w2v, pos1, pos2], dim=-1).unsqueeze(0).unsqueeze(0)
                feature = self.conv(feature).squeeze(-1)

                feature = F.max_pool1d(feature, feature.size(-1)).squeeze(-1) + self.conv_bias
                # this tanh is little different from lin-16's.
                feature = self.tanh(feature)
                feature = self.dropout(feature)
                # dropout is a little different too.
                features.append(feature)
            features = torch.cat(features, dim=0)

            atten_weights = self.atten_sm(torch.bmm(self.r_embed.unsqueeze(1),
                                                    torch.matmul(self.att_W, features.t())).squeeze(1))
            # to this point, features is actually s
            # n_rel * D
            features = torch.matmul(atten_weights, features)
            if self.dropout is not None:
                features = self.dropout(features)
                if not self.training:
                    features = features * 0.5

            batch_features.append(features.unsqueeze(0))
        return torch.cat(batch_features, dim=0)





