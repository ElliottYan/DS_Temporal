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
from cnn import CNN
from cnn_word_mem import CNN_WORD_MEM


class TEMP_MEM(nn.Module):
    def __init__(self, settings):
        super(TEMP_MEM, self).__init__()
        self.conv_type = settings['conv_type']
        if settings['use_word_mem']:
            self.enc = CNN_WORD_MEM(settings)
        elif self.conv_type == 'CNN':
            self.enc = CNN(settings)
        else:
            self.enc = PCNN(settings)

        self.r_embed = nn.Parameter()

    def forward(self, input):
        conv_out = self.enc(input)


    # query_type : r_embed
    def _create_queries(self, inputs, encoding_output=None):
        batch_size = len(inputs)
        # without entity queries
        # queries = self.r_embed.expand((batch_size, ) + self.r_embed.size())
        # queries = []
        # for _ in range(batch_size):
        #     queries.append(self.r_embed)
        queries = torch.stack([self.r_embed] * batch_size)
        return queries
