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
        self.out_linear = nn.Linear(self.out_feature_size, 1)

    def test_fusion(self, conv_out):
        # batch_size * n_rel * out_feature_size
        att_feature = self._merge_each_rel_rep(conv_out).squeeze()
        out_feature = self.r_mem(att_feature)
        # out_feature = torch.matmul(out_feature, self.r_embed.t()).squeeze() + self.r_bias
        out_feature = self.out_linear(out_feature).squeeze(-1)
        return F.sigmoid(out_feature)

    def train_fusion(self, conv_out, labels):
        att_feature = self._merge_each_rel_rep(conv_out).squeeze()
        out_feature = self.r_mem(att_feature)
        out_feature = self.out_linear(out_feature).squeeze(-1)
        return out_feature



