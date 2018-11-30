import sys
import torch
import pdb
import math
import torch.nn as nn
from cnn import CNN
from pcnn import PCNN

class MIML_CONV(nn.Module):
    def __init__(self, settings):
        super().__init__()
        '''
        self.conv = CNN(settings)
        self.linear = nn.Linear(settings['out_c'], settings['n_rel'])
        '''
        # for PCNN
        self.conv = PCNN(settings)
        self.linear = nn.Linear(settings['out_c'] * 3, settings['n_rel'])
        # self.pred_sm = nn.Sigmoid()
        self.pred_sm = nn.LogSigmoid()
        self.dropout = nn.Dropout(p=settings['dropout_p'])

    def forward(self, inputs):
        conv_out = self.conv(inputs)
        # pdb.set_trace()
        cross_sent_max = [torch.max(item, 0, keepdim=True)[0] for item in conv_out]
        cross_sent_feature = torch.cat(cross_sent_max, dim=0)
        out_feature = self.dropout(cross_sent_feature)
        out_feature = self.linear(out_feature)
        # out_feature = self.pred_sm(out_feature)
        return out_feature



class MIML_CONV_ATT(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.conv = CNN(settings)
        self.n_rel = settings['n_rel']
        self.out_c = settings['out_c']
        self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.out_c), requires_grad=True)
        self.r_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)

        # attention module
        self.att_sm = nn.Softmax(dim=-1)
        eye = torch.eye(self.out_c, self.out_c).unsqueeze(0)
        # self.att_W = nn.Parameter(eye.expand(self.n_rel, self.out_c, self.out_c), requires_grad=True)
        self.att_W = nn.Parameter(eye.repeat([self.n_rel, 1, 1]))
        # pcnn
        # self.linear = nn.Linear(settings['out_c'] * 3, settings['n_rel'])
        self.linear = nn.Linear(settings['n_rel'], settings['n_rel'])
        self.dropout = nn.Dropout(p=settings['dropout_p'])

        con = math.sqrt(6.0/(self.out_c + self.n_rel))
        nn.init.uniform_(self.r_embed, a=-con, b=con)
        nn.init.uniform_(self.r_bias, a=-con, b=con)


    def forward(self, inputs):
        bz = len(inputs)
        conv_out = self.conv(inputs)

        att_outs = []
        for each_conv_out in conv_out:
            atten_weights = self.att_sm(torch.bmm(self.r_embed.unsqueeze(1),
                                                   torch.matmul(self.att_W, each_conv_out.t())).squeeze(1))
            # n_rel * out_c
            att_feature = torch.matmul(atten_weights, each_conv_out)
            att_outs.append(att_feature)
        # bz * n_rel * out_c
        att_feature = torch.stack(att_outs)
        # modified the computing of scores
        # scores = torch.matmul(att_feature, self.r_embed.t()).squeeze() + self.r_bias
        scores = (att_feature * (self.r_embed.unsqueeze(0).repeat([bz, 1, 1]))).sum(dim=-1) + self.r_bias
        # n_rel * D
        out_feature = self.dropout(scores)
        # out_feature = self.linear(out_feature)
        return out_feature



