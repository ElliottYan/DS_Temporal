import sys
import torch
import pdb
import math
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN
from cnn_word_mem import CNN_WORD_MEM
from pcnn import PCNN

class MIML_CONV(nn.Module):
    def __init__(self, settings):
        super().__init__()
        if settings['conv_type'] == 'CNN':
            # for CNN
            self.enc = CNN(settings)
            self.linear = nn.Linear(settings['out_c'], settings['n_rel'])
        else:
            # for PCNN
            self.enc = PCNN(settings)
            self.linear = nn.Linear(settings['out_c'] * 3, settings['n_rel'])
        # self.pred_sm = nn.Sigmoid()
        self.pred_sm = nn.LogSigmoid()
        self.dropout = nn.Dropout(p=settings['dropout_p'])

    def forward(self, inputs):
        conv_out = self.enc(inputs)
        # pdb.set_trace()
        cross_sent_max = [torch.max(item, 0, keepdim=True)[0] for item in conv_out]
        cross_sent_feature = torch.cat(cross_sent_max, dim=0)
        out_feature = self.dropout(cross_sent_feature)
        out_feature = self.linear(out_feature)
        # out_feature = self.pred_sm(out_feature)
        if not self.training:
            # need to compute probability for testing.
            out_feature = F.sigmoid(out_feature)
        return out_feature



class MIML_CONV_ATT(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.enc_type = settings['conv_type']
        self.out_c = settings['out_c']
        if self.enc_type == 'CNN':
            self.enc = CNN(settings)
        else:
            self.enc = PCNN(settings)
            self.out_c *= 3
        self.out_feature_size = self.out_c

        self.n_rel = settings['n_rel']
        self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.out_feature_size), requires_grad=True)
        self.r_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)

        # attention module
        self.att_sm = nn.Softmax(dim=-1)
        eye = torch.eye(self.out_feature_size, self.out_feature_size)
        # self.att_W = nn.Parameter(eye.expand(self.n_rel, self.out_c, self.out_c), requires_grad=True)
        # n_rel * out_feature_size * out_c
        self.att_W = nn.Parameter(eye.unsqueeze(0).repeat([self.n_rel, 1, 1]), requires_grad=True)
        # out_feature_size * out_c
        self.att_W_small = nn.Parameter(eye, requires_grad=True)
        # pcnn
        # self.linear = nn.Linear(settings['out_feature_size'] * 3, settings['n_rel'])
        self.linear = nn.Linear(self.out_feature_size, 1)
        self.dropout = nn.Dropout(p=settings['dropout_p'])

        # con = math.sqrt(6.0/(self.out_feature_size + self.n_rel))
        con = 0.01
        nn.init.uniform_(self.r_embed, a=-con, b=con)
        nn.init.uniform_(self.r_bias, a=-con, b=con)
        nn.init.uniform_(self.att_W_small, a=-con, b=con)


    def forward(self, inputs):
        labels = [item['label'] for item in inputs]

        bz = len(inputs)
        conv_out = self.enc(inputs)
        if self.training:
            out_feature = self.train_fusion(conv_out, labels)
        else:
            out_feature = self.test_fusion(conv_out)
        return out_feature

    def train_fusion(self, conv_out, labels):
        att_outs = []
        for ix, each_conv_out in enumerate(conv_out):
            r_embed = self.r_embed[labels[ix]].sum(-2, keepdim=True)
            # r_embed = self.r_embed[labels[ix]]
            atten_weights = self.att_sm(torch.matmul(r_embed, torch.matmul(self.att_W_small, each_conv_out.t())))

            # atten_weights = self.att_sm(torch.bmm(self.r_embed.unsqueeze(1),
            #                                        torch.matmul(self.att_W, each_conv_out.t())).squeeze(1))
            # n_rel * out_feature_size
            att_feature = torch.matmul(atten_weights, each_conv_out)
            att_outs.append(att_feature)
        # bz * out_feature_size
        att_feature = torch.cat(att_outs, dim=0)
        # modified the computing of scores
        # n_rel * D
        out_feature = self.dropout(att_feature)
        # out_feature = self.linear(out_feature)
        out_feature = torch.matmul(out_feature, self.r_embed.t()) + self.r_bias
        out_feature = out_feature.squeeze()
        return out_feature

    def test_fusion(self, conv_out):
        att_outs = []
        for ix, each_conv_out in enumerate(conv_out):
            # n_rel * out_feature_size * out_c
            expand_att_W = self.att_W_small.unsqueeze(0).repeat([self.n_rel, 1, 1])
            # bilinear
            z = torch.bmm(self.r_embed.unsqueeze(1), torch.matmul(expand_att_W, each_conv_out.t()).squeeze(1))
            atten_weights = self.att_sm(z)
            # n_rel * out_feature_size
            att_feature = torch.matmul(atten_weights, each_conv_out)
            att_outs.append(att_feature)

        # bz * n_rel * out_feature_size
        att_feature = torch.stack(att_outs)
        # modified the computing of scores
        out_feature = torch.matmul(att_feature, self.r_embed.t()).squeeze() + self.r_bias
        out_feature = out_feature.max(1)[0]
        # needed for testing...
        # out_feature = F.softmax(out_feature, -1)
        out_feature = F.sigmoid(out_feature)
        return out_feature

class MIML_CONV_WORD_MEM_ATT(MIML_CONV_ATT):
    def __init__(self, settings):
        super(MIML_CONV_WORD_MEM_ATT, self).__init__(settings)
        self.enc = CNN_WORD_MEM(settings)
        self.out_feature_size += self.enc.input_size
        # re-define the attention parameter. Need to clean this logic later.
        self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.out_feature_size), requires_grad=True)
        self.r_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)
        eye = torch.eye(self.out_feature_size, self.out_feature_size)
        self.att_W_small = nn.Parameter(eye, requires_grad=True)
        con = 0.01
        nn.init.uniform_(self.r_embed, a=-con, b=con)
        nn.init.uniform_(self.r_bias, a=-con, b=con)
        nn.init.uniform_(self.att_W_small, a=-con, b=con)






