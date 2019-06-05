import torch
import torch.nn as nn
import pytorch_pretrained_bert as ppb

import pdb
import math

import logging
logging.basicConfig(level=logging.INFO)

class BERT_FINE_TUNE(nn.Module):
    """
    A wrapper for ppb.BertModel fine tuning.
    """
    def __init__(self, settings):
        super(BERT_FINE_TUNE, self).__init__()
        self.model_dir = './origin_data/pretrained_weights/bert-base-uncased'
        self.bert = ppb.BertModel.from_pretrained(self.model_dir)

        self.word_padding_idx = 0
        use_cuda = settings['use_cuda']
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.n_rel = settings['n_rel']
        self.linear = nn.Linear(768, self.n_rel)
        self.pred_sm = nn.LogSoftmax()

    def forward(self, inputs):
        bz = len(inputs)
        labels = [item['label'] for item in inputs]
        bag_sizes = [len(bag['bag']) for bag in inputs]
        sents = []

        # todo : for now we just use word embeddings and ignore the position feature.
        # print('process sentences.')
        for bag in inputs:
            for sent in bag['bag']:
                # the sentence inputs
                sents.append(sent[0])

        # padding.
        # already have cls token at the start.
        sents = nn.utils.rnn.pad_sequence(sents, padding_value=self.word_padding_idx, batch_first=False).to(self.device)
        segments = torch.zeros_like(sents, device=self.device)

        # out : L * N_sents * D
        '''
        out = self.transformer(sents, lengths=sents_length)
        out = out[1]
        '''

        # split batch to avoid oom.
        # this is used when oom happens.
        num_batch = math.ceil(sents.shape[1] / bz)
        out_list = []
        with torch.no_grad():
            for i in range(num_batch):
                encoded_layers, _ = self.bert(sents[:, i*bz: (i+1)*bz], segments[:, i*bz:(i+1)*bz])
                # how to ensemble encoded_layers
                encoded_layers = sum(encoded_layers)
                # only choose the rep on cls token
                out_list.append(encoded_layers[0])

        out = torch.cat([item for item in out_list], dim=0) # N_sents * D
        out = self.linear(out)  # N_sents * n_rel

        start = 0
        features = []
        for item in bag_sizes:
            features.append(out[start:start + item])
            start += item

        # take max, at the first place.
        batch_features = []
        for ix in range(bz):
            label = labels[ix]
            feature = features[ix]
            feature = self.pred_sm(feature)
            if self.training:
                feature = feature[torch.max(feature, dim=0)[1][label]].contiguous().view(1, -1)
            else:
                feature = torch.max(feature, dim=0)[0].view(1, -1)
            batch_features.append(feature)
        pred = torch.cat(batch_features, dim=0)

        '''
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

        '''
        return pred


