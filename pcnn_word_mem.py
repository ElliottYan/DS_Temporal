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
from pcnn import PCNN
from word_rel_mem import Word_MEM


class PCNN_WORD_MEM(PCNN):
    def __init__(self, settings):
        super(PCNN_WORD_MEM, self).__init__(settings)
        self.word_mem = Word_MEM(self.input_size, settings)

    def _enc_each_iter(self, feature, item):
        conv_feature = super(PCNN_WORD_MEM, self)._enc_each_iter(feature, item)
        word_mem_feature = self.word_mem({
            'w2v': feature.reshape(-1, self.input_size),
            'sent': item,
        }).reshape(1, -1)  # 1 * feature_size

        ret_feature = torch.cat([conv_feature, word_mem_feature], dim=-1)
        return ret_feature

