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

# from dataset import Dataset, Temporal_Data
# from dataset import collate_fn, collate_fn_temporal_for_pcnn, collate_fn_temporal, collate_fn1
# import numpy as np
# from character_process import n_letters
# from utils import pad_sequence


class CNN_ONE(nn.Module):
    def __init__(self, settings):
        super(CNN_ONE, self).__init__()
        self.word_embed_size = settings['word_embed_size']
        self.pos_embed_size = settings['pos_embed_size']
        self.input_size = self.word_embed_size + 2 * self.pos_embed_size
        self.out_c = settings['out_c']
        self.window = 3
        self.n_rel = settings['n_rel']
        self.vocab_size = settings['vocab_size']
        self.pos_limit = settings['pos_limit']

        self.conv = nn.Conv2d(1, self.out_c, (self.window, self.input_size), padding=(self.window-1, 0), bias=False)
        self.conv_bias = nn.Parameter(torch.zeros(1, self.out_c),requires_grad=True)

        self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.out_c), requires_grad=True)
        self.r_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.pred_sm = nn.LogSoftmax(dim=-1)

        self.w2v = nn.Embedding(self.vocab_size, self.word_embed_size)
        self.pos1_embed = nn.Embedding(self.pos_limit * 2 + 1, self.pos_embed_size)
        self.pos2_embed = nn.Embedding(self.pos_limit * 2 + 1, self.pos_embed_size)
        # pretrained embedding
        self.w2v.weight = nn.Parameter(torch.FloatTensor(settings['word_embeds']), requires_grad=True)

        # init
        con = math.sqrt(6.0/(self.out_c + self.n_rel))
        con1 = math.sqrt(6.0 / ((self.pos_embed_size + self.word_embed_size)*self.window))
        nn.init.uniform_(self.conv.weight, a=-con1, b=con1)
        nn.init.uniform_(self.conv_bias, a=-con1, b=con1)
        nn.init.uniform_(self.r_embed, a=-con, b=con)
        nn.init.uniform_(self.r_bias, a=-con, b=con)

    def forward(self, input):
        bags = [item['bag'] for item in input]
        labels = [item['label'] for item in input]
        preds = self._create_sentence_embedding(bags, labels)
        return torch.cat(preds, dim=0)

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
                feature = torch.matmul(feature, self.r_embed.t())
                # dropout is a little different too.
                if not self.training:
                    feature *= 0.5
                feature += self.r_bias
                features.append(feature)
            features = torch.cat(features, dim=0) # N_sents * n_rel
            features = self.pred_sm(features)
            if self.training:
                features = features[torch.max(features, dim=0)[1][label]]
            else:
                features = torch.max(features, dim=0)[0].view(1, -1)

            batch_features.append(features)
        return batch_features





