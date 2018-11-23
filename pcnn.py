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
# torch.cuda.manual_seed(1)
# torch.manual_seed(1)

class PCNN(nn.Module):
    def __init__(self, settings):
        super(PCNN, self).__init__()
        self.word_embed_size = settings['word_embed_size']
        self.pos_embed_size = settings['pos_embed_size']
        self.input_size = self.word_embed_size + 2 * self.pos_embed_size
        self.out_c = settings['out_c']
        self.window = 3
        self.n_rel = settings['n_rel']
        self.vocab_size = settings['vocab_size']
        self.pos_limit = settings['pos_limit']

        # torch.cuda.manual_seed(2)
        # torch.manual_seed(2)

        self.conv = nn.Conv1d(1, self.out_c, (self.window, self.input_size), padding=(self.window//2, 1-self.window//2), bias=False)
        self.feature_size = 3 * self.out_c
        self.conv_bias_0 = nn.Parameter(torch.zeros(1, self.out_c),requires_grad=True)
        self.conv_bias_1 = nn.Parameter(torch.zeros(1, self.out_c),requires_grad=True)
        self.conv_bias_2 = nn.Parameter(torch.zeros(1, self.out_c),requires_grad=True)

        self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.feature_size), requires_grad=True)
        self.r_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(settings['dropout_p'])
        self.pred_sm = nn.LogSoftmax(dim=-1)
        self.atten_sm = nn.Softmax(dim=-1)
        self.limit = 30

        self.w2v = nn.Embedding(self.vocab_size, self.word_embed_size)
        self.pos1_embed = nn.Embedding(self.pos_limit * 2 + 1, self.pos_embed_size)
        self.pos2_embed = nn.Embedding(self.pos_limit * 2 + 1, self.pos_embed_size)
        # pretrained embedding
        self.w2v.weight = nn.Parameter(torch.FloatTensor(settings['word_embeds']), requires_grad=True)

        eye = torch.eye(self.feature_size, self.feature_size)
        self.att_W = nn.Parameter(eye.expand(self.n_rel, self.feature_size, self.feature_size), requires_grad=True)

        # init
        con = math.sqrt(6.0/(self.out_c + self.n_rel))
        con1 = math.sqrt(6.0 / ((self.pos_embed_size + self.word_embed_size)*self.window))
        nn.init.uniform_(self.conv.weight, a=-con1, b=con1)
        nn.init.uniform_(self.conv_bias_0, a=-con1, b=con1)
        nn.init.uniform_(self.conv_bias_1, a=-con1, b=con1)
        nn.init.uniform_(self.conv_bias_2, a=-con1, b=con1)
        nn.init.uniform_(self.r_embed, a=-con, b=con)
        nn.init.uniform_(self.r_bias, a=-con, b=con)

    def forward(self, input):
        bags = [item['bag'] for item in input]
        labels = [item['label'] for item in input]
        features = self._create_sentence_embedding(bags, labels)
        return features

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
                pos1_mask = item[:, 1] >= self.limit
                pos2_mask = item[:, 2] >= self.limit
                left = pos1_mask * pos2_mask
                right = (1-pos1_mask) * (1-pos2_mask)
                # mid = torch.abs(pos1_mask - pos2_mask)
                mid = 1 - right - left
                left_feature = F.max_pool1d(feature * left.float(), feature.size(-1)).squeeze(-1) + self.conv_bias_0
                mid_feature = F.max_pool1d(feature * mid.float(), feature.size(-1)).squeeze(-1) + self.conv_bias_1
                right_feature = F.max_pool1d(feature * right.float(), feature.size(-1)).squeeze(-1) + self.conv_bias_2
                feature = torch.cat([left_feature, mid_feature, right_feature]).reshape(1, self.feature_size)
                # this tanh is little different from lin-16's.
                feature = self.tanh(feature)
                feature = self.dropout(feature)
                # dropout is a little different too.
                features.append(feature)
            features = torch.cat(features, dim=0)
            batch_features.append(features)
        return batch_features





