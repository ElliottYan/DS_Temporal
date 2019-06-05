"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np
import math
import pdb

from transformer.encoders.transformer_encoder import TransformerEncoder
from transformer.modules import Embeddings

from utils import logging_existing_tensor


class TRANSFORMER_ENCODER(nn.Module):
    def __init__(self, settings):
        super(TRANSFORMER_ENCODER, self).__init__()
        num_layers = settings['num_layers']
        d_model = settings['d_model']
        heads = settings['heads']
        d_ff = settings['d_ff']
        dropout = settings['dropout_p']
        max_relative_positions = settings['max_relative_positions']
        self.vocab_size = settings['vocab_size']
        self.word_embed_size = settings['word_embed_size']
        # cuda semantics
        self.use_cuda = settings['use_cuda']
        if self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        '''
        # default to use pretrain embeddings.
        user_pretrain = True
        if user_pretrain:
            embed = np.concatenate([settings['word_embeds'], np.random.normal(size=(1, self.word_embed_size))])
            self.w2v = nn.Embedding.from_pretrained(torch.FloatTensor(embed), freeze=False)
            # in pytorch 1.0, from_pretrain method doesn't support set padding_idx in init function.
            self.w2v.padding_idx = self.vocab_size
        else:
            # randomly initialized.
            self.w2v = nn.Embedding(self.vocab_size + 1, self.word_embed_size, padding_idx=self.vocab_size)
        # for compatibility in TransformerEncoder.
        self.w2v.word_padding_idx = self.w2v.padding_idx
        '''

        # Embeddings using onmt modules.
        position_encoding = True
        # for feat embeddings
        feat_merge = 'concat'
        feat_vec_exponent = 0.7
        feat_vec_size = -1
        feat_pad_indices = []
        num_feat_embeddings = []

        # add two special tokens
        extra_tokens = ['cls', 'padding']
        self.extra_tokens_ids = {extra_tokens[i]: self.vocab_size + i for i in range(len(extra_tokens))}
        num_word_embeddings = self.vocab_size + len(extra_tokens)
        word_padding_idx = self.extra_tokens_ids['padding']
        self.emb = Embeddings(
            word_vec_size=d_model,
            position_encoding=position_encoding,
            feat_merge=feat_merge,
            feat_vec_exponent=feat_vec_exponent,
            feat_vec_size=feat_vec_size,
            dropout=dropout,
            word_padding_idx=word_padding_idx,
            feat_padding_idx=feat_pad_indices,
            word_vocab_size=num_word_embeddings,
            feat_vocab_sizes=num_feat_embeddings,
        )

        use_pretrain = settings['use_pretrain_embedding']
        # load pretrained vectors.
        if use_pretrain:
            word_embeds = settings['word_embeds']
            if word_embeds.shape[-1] != d_model:
                raise ValueError("When use pretrain embedding, d_model must be the same with pretrained embedding dim.")
            pretrained = create_embedding_matrix(word_embeds, extra_tokens)
            self.emb.word_lut.weight.data.copy_(pretrained)

        self.transformer = TransformerEncoder(num_layers,
                                              d_model,
                                              heads,
                                              d_ff,
                                              dropout,
                                              embeddings=self.emb,
                                              max_relative_positions=max_relative_positions)

        self.feature_size = d_model
        self.n_rel = settings['n_rel']
        # use for attention
        eye = torch.eye(self.feature_size, self.feature_size)
        self.att_W = nn.Parameter(eye.expand(self.n_rel, self.feature_size, self.feature_size), requires_grad=True)
        self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.feature_size), requires_grad=True)
        self.r_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)
        self.pred_sm = nn.LogSoftmax(dim=-1)
        self.atten_sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(settings['dropout_p'])

        # self.cnt = 0

    def forward(self, inputs):
        # logging_existing_tensor()

        bz = len(inputs)
        labels = [item['label'] for item in inputs]
        bag_sizes = [len(bag['bag']) for bag in inputs]
        sents = []
        sents_length = []
        # todo : for now we just use word embeddings and ignore the position feature.
        # print('process sentences.')
        for bag in inputs:
            for sent in bag['bag']:
                # tmp_sent = sent[:, 0]
                # sents.append(tmp_sent)
                # sents_length.append(len(tmp_sent))
                sents.append(sent[:, 0])
                sents_length.append(len(sent[:, 0]))

        # padding.
        sents = nn.utils.rnn.pad_sequence(sents, padding_value=self.emb.word_padding_idx, batch_first=False).unsqueeze(-1)

        # add cls token
        tmp_shape = list(sents.shape)
        tmp_shape[0] = 1
        sents = torch.cat([torch.ones(tmp_shape, device=self.device).long() * self.extra_tokens_ids['cls'], sents], dim=0)
        sents_length = [item + 1 for item in sents_length]

        sents_length = torch.LongTensor(sents_length).to(self.device)

        # out : L * N_sents * D
        '''
        out = self.transformer(sents, lengths=sents_length)
        out = out[1]
        '''
        # split batch to avoid oom.
        # this is used when oom happens.
        num_batch = math.ceil(sents.shape[1] / bz)
        out_list = []
        for i in range(num_batch):
            out_list.append(self.transformer(sents[:, i*bz: (i+1)*bz], lengths=sents_length[i*bz: (i+1)*bz]))
        out = torch.cat([item[1] for item in out_list], dim=1)


        # emb, out, lengths
        # takes the output at cls token.
        out = out[0]     # N_sents * D

        start = 0
        features = []
        for item in bag_sizes:
            features.append(out[start:start + item].contiguous().reshape(-1, self.feature_size))
            start += item

        # print('fusion')
        s = self.fusion(features)

        # print('merge')
        if self.training:
            idx = []
            for cnt, item in enumerate(labels):
                idx += [cnt] * len(item)
            # B' * n_rel * D
            tmp = s[torch.tensor(idx, device=self.device, requires_grad=False).long()]
            # B' * D
            s = tmp[torch.arange(0, tmp.size(0), device=self.device).long(), torch.cat(labels, dim=0)]
            # score is the same, but normalize over different set!
            scores = torch.matmul(s, self.r_embed.t()) + self.r_bias
            pred = self.pred_sm(scores)
        else:
            scores = torch.matmul(s, self.r_embed.t()) + self.r_bias
            pred = self.pred_sm(scores.view(-1, self.n_rel)).view(bz, self.n_rel, self.n_rel).max(1)[0]

        return pred

    '''
    def forward(self, inputs):
        outs = []
        for bag in inputs:
            for sent in bag['bag']:
                out = self.transformer(sent.unsqueeze(-1))
                outs.append(out)
        return outs
    '''

    def fusion(self, features):
        ret = []
        for feature in features:
            atten_weights = self.atten_sm(torch.bmm(self.r_embed.unsqueeze(1),
                                                    torch.matmul(self.att_W, feature.t())).squeeze(1))
            # to this point, features is actually s
            # n_rel * D
            feature = torch.matmul(atten_weights, feature)
            if self.dropout is not None:
                feature = self.dropout(feature)
                if not self.training:
                    feature = feature * 0.5
            ret.append(feature)
        return torch.stack(ret)



def create_embedding_matrix(matrix, extra_tokens):
    """
    Add extra tokens to embedding matrix and convert to torch.FloatTensor
    :param matrix: pretrained word embeddings, e.g. Glove
    :param extra_tokens: contains extra tokens to be added, in order.
    :return: the converted tensor with extra token tensor appended at the end.
    """
    vocab_size = matrix.shape[0]
    word_embed_size = matrix.shape[-1]

    for token in extra_tokens:
        token_tensor = np.random.normal(size=(1, word_embed_size))
        matrix = np.concatenate([matrix, token_tensor])
        vocab_size += 1
    out = torch.FloatTensor(matrix)
    return out




