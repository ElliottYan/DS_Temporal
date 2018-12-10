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
from miml_conv import MIML_CONV_ATT

from word_rel_mem import Rel_MEM

class MIML_CONV_ATT_REL_MEM(MIML_CONV_ATT):
    def __init__(self, settings):
        super(MIML_CONV_ATT_REL_MEM, self).__init__(settings)
        self.r_mem = Rel_MEM(self.out_feature_size, settings)
        self.m_embed = nn.Parameter(torch.zeros(self.n_rel, self.out_feature_size))
        nn.init.uniform_(self.m_embed.data, -0.01, 0.01)

    def test_fusion(self, conv_out):
        # batch_size * n_rel * out_feature_size
        att_feature = self._merge_each_rel_rep(conv_out).squeeze()
        out_feature = self.r_mem(att_feature)
        out_feature = torch.matmul(out_feature, self.m_embed.t()).squeeze() + self.r_bias
        out_feature = out_feature.max(1)[0]
        return F.sigmoid(out_feature)

    def train_fusion(self, conv_out, labels):
        att_feature = self._merge_label_rel_rep(conv_out, labels)
        # modified the computing of scores
        # n_rel * D
        out_feature = self.dropout(att_feature)
        # out_feature = self.linear(out_feature)
        out_feature = torch.matmul(out_feature, self.m_embed.t()) + self.r_bias
        out_feature = out_feature.squeeze()
        return out_feature



