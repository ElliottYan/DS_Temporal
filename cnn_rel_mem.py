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

from word_rel_mem import Word_MEM, Rel_MEM

class CNN_REL_MEM(MIML_CONV_ATT):
    def __init__(self, settings):
        super(CNN_REL_MEM, self).__init__(settings)
        self.r_mem = Rel_MEM(settings, self.out_feature_size)

    def train_fusion(self, conv_out, labels):
        # batch_size * n_rel * out_feature_size
        att_feature = self._merge_each_rel_rep(conv_out)
        out_feature = self.r_mem(att_feature)
        return out_feature

    def test_fusion(self, conv_out):
        return self.train_fusion(conv_out, None)


