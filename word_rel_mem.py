import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from mem_cnn import AttrProxy
from Dataset import Riedel_10, WIKI_TIME
import pdb
import math
from cnn import CNN
from pcnn import PCNN

class Word_MEM(nn.Module):
    def __init__(self, word_embed_size, settings):
        super(Word_MEM, self).__init__()
        self.max_hops = settings['word_mem_hops']
        self.word_embed_size = word_embed_size

        for i in range(self.max_hops):
            # also add bias ?
            '''
            if i == 0:
                C = nn.Linear(2 * self.word_embed_size , self.word_embed_size, bias=False)
            else:
            '''
            C = nn.Linear(self.word_embed_size, self.word_embed_size, bias=False)
            C.weight.data.uniform_(-0.01, 0.01)
            self.add_module('C_{}'.format(i), C)
        self.C = AttrProxy(self, 'C_')

        # self.step_through = nn.Linear(2 * self.word_embed_size, 2 * self.word_embed_size)
        self.scoring_1 = nn.Linear(3 * self.word_embed_size, 1)
        self.scoring_2 = nn.Linear(2 * self.word_embed_size, 1)
        nn.init.uniform_(self.scoring_2.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.scoring_2.bias, a=-0.01, b=0.01)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, input):
        # word_embed = self.
        w2v = input['w2v']
        sent = input['sent']
        query, memory = split_query_and_memory(w2v, sent)

        # first version
        for hop in range(self.max_hops):
            if not hop:
                # sum two queries
                query = query.contiguous().view(2, -1).sum(0, True)
            _query = torch.cat([query] * memory.shape[0], dim=0)  # L_memory * 2d or L_memory * d
            # if not hop:
            #     g = self.tanh(self.scoring_1(torch.cat([memory, _query], dim=-1)))  # L_memory * 1
            # else:
            g = self.tanh(self.scoring_2(torch.cat([memory, _query], dim=-1)))  # L_memory * 1
            weights = self.softmax(g.reshape(1, -1))  # 1 * L_memory
            x = torch.matmul(weights, memory)  # 1 * d
            # if not hop:
            #     query = self.C[0](query) + x
            # else:
            query = self.C[0](query) + x

        '''
        # maybe the two entity vectors are used as query independently.
        for hop in range(self.max_hops):
            s = query.shape
            _query = query.unsqueeze(1).reshape([s[0], 2, -1])
        '''


        return query

class Rel_MEM(nn.Module):
    def __init__(self, r_feature_size, settings):
        super(Rel_MEM, self).__init__()
        self.max_hops = settings['rel_mem_hops']
        """
        for i in range(self.max_hops):
            # also add bias ?
            C = nn.Linear(r_feature_size, r_feature_size, bias=False)
            C.weight.data.uniform_(-0.01, 0.01)
            self.add_module('C_{}'.format(i), C)
        self.C = AttrProxy(self, 'C_')
        """
        self.att_W = nn.Parameter(torch.randn(r_feature_size, r_feature_size), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        # use different r_embed here...
        self.r_embed = nn.Parameter(torch.zeros(settings['n_rel'], r_feature_size), requires_grad=True)
        # add q, kv
        self.q_embed = nn.Parameter(torch.zeros(r_feature_size, r_feature_size), requires_grad=True)
        self.kv_embed = nn.Parameter(torch.zeros(r_feature_size, r_feature_size), requires_grad=True)

        # init
        self.att_W.data.uniform_(-0.01, 0.01)
        self.r_embed.data.uniform_(-0.01, 0.01)

        self.q_embed.data.uniform_(-0.01, 0.01)
        self.kv_embed.data.uniform_(-0.01, 0.01)


    def forward(self, r_reps):
        """
        :param r_reps: B * n_rel * r_feature_size
        :return:
        """
        version = 6
        # print('REL MEM Version : {}'.format(version))
        for hop in range(self.max_hops):
            if version == 1:
                r_reps = self._hop_computation_v1(r_reps)
            elif version == 2:
                r_reps = self._hop_computation_v2(r_reps)
            elif version == 3:
                r_reps = self._hop_computation_v3(r_reps)
            elif version == 4:
                r_reps = self._hop_computation_v4(r_reps)
            elif version == 5:
                r_reps = self._hop_computation_v5(r_reps)
            elif version == 6:
                r_reps = self._hop_computation_v6(r_reps)
        return r_reps

    def _hop_computation_v1(self, r_reps):
        """
        q, k, v : r_reps
        not work...
        """
        tmp = torch.matmul(r_reps, self.att_W)
        tmp = torch.bmm(r_reps, torch.transpose(tmp, 1, 2))
        # B * n_rel * n_rel
        atten_weights = self.softmax(tmp)
        # atten_weights = self.softmax(
        #                     torch.bmm(r_reps,
        #                               torch.transpose(r_reps, 1, 2)))  # B * n_rel * n_rel
        r_reps = torch.bmm(atten_weights, r_reps)
        return r_reps

    def _hop_computation_v2(self, r_reps):
        """
        q : r_embed
        k, v : r_reps
        """
        tmp = torch.matmul(self.r_embed, self.att_W)
        tmp = torch.matmul(r_reps, torch.transpose(tmp, 0, 1))
        atten_weights = self.softmax(tmp)
        r_reps = torch.bmm(atten_weights, r_reps)
        return r_reps

    def _hop_computation_v3(self, r_reps):
        """
        q : r_reps
        k, v : r_embed
        """
        tmp = torch.matmul(self.r_embed, self.att_W)
        tmp = torch.matmul(r_reps, torch.transpose(tmp, 0, 1))
        # softmax should be operated on key level.
        tmp = torch.transpose(tmp, 1, 2)
        atten_weights = self.softmax(tmp)
        r_reps = torch.matmul(atten_weights, self.r_embed)
        return r_reps

    def _hop_computation_v4(self, r_reps):
        """
        q, k, v : r_reps
        maps to different feature space.
        """
        q_reps = torch.matmul(r_reps, self.q_embed)
        kv_reps = torch.matmul(r_reps, self.kv_embed)
        tmp = torch.matmul(q_reps, self.att_W)
        tmp = torch.bmm(kv_reps, torch.transpose(tmp, 1, 2))
        # B * n_rel * n_rel
        atten_weights = self.softmax(tmp)
        r_reps = torch.bmm(atten_weights, kv_reps)
        return r_reps

    def _hop_computation_v5(self, r_reps):
        """
        q : labeled r_reps
        k, v : rel_embed
        not work...
        """
        tmp = torch.matmul(self.r_embed, self.att_W)
        tmp = torch.matmul(r_reps, torch.transpose(tmp, 0, 1))
        # softmax should be operated on key level.
        tmp = torch.transpose(tmp, 1, 2)
        atten_weights = self.softmax(tmp)
        r_reps = torch.matmul(atten_weights, self.r_embed)
        return r_reps

    def _hop_computation_v6(self, r_reps):
        """
        q, k, v : r_reps
        r_rel only for testing. remove the att_W which is not trained...
        """
        tmp = torch.bmm(r_reps, torch.transpose(r_reps, 1, 2))
        # B * n_rel * n_rel
        atten_weights = self.softmax(tmp)
        r_reps = torch.bmm(atten_weights, r_reps)
        return r_reps


class Word_Rel_MEM(nn.Module):
    def __init__(self, settings):
        super(Word_Rel_MEM, self).__init__()
        # Read-in configs
        self.use_cuda = settings['use_cuda']
        self.cuda_devices = [4, 5, 6, 7]

        self.word_embed_size = settings['word_embed_size']
        self.vocab_size = settings['vocab_size']
        self.n_rel = settings['n_rel']
        self.hidden_size = 50
        self.features_size = settings['word_embed_size']
        self.tri_attention = settings['tri_attention']


        self.pos_embed_size = 5
        self.features_size += 2 * self.pos_embed_size
        # define the position embedding effective domain
        # self.max_len = 60
        self.pos_limit = settings['pos_limit']
        # number of output channels for CNN
        self.out_c = settings['out_c']
        self.sent_feat_size = self.out_c
        self.dropout_p = settings['dropout_p']
        pre_word_embeds = settings['word_embeds']
        self.version = settings['version']
        self.remove_origin_query = settings['remove_origin_query']

        # word embed
        self.w2v = nn.Embedding(self.vocab_size, self.word_embed_size, padding_idx=self.vocab_size-1)
        # pre-trained
        if pre_word_embeds is not None:
            self.pre_word_embed = True
            self.w2v.weight.data[:pre_word_embeds.shape[0]] = nn.Parameter(torch.FloatTensor(pre_word_embeds), requires_grad=True)
        else:
            self.pre_word_embed = False

        # pos embed
        self.position_embedding = settings['position_embedding']
        if settings['position_embedding']:
            self.pos1_embed = nn.Embedding(self.pos_limit * 2 + 1, self.pos_embed_size)
            self.pos2_embed = nn.Embedding(self.pos_limit * 2 + 1, self.pos_embed_size)
        else:
            self.features_size -= 2 * self.pos_embed_size

        self.window = 3
        self.conv = nn.Conv2d(1, self.out_c, (self.window, self.features_size), padding=(self.window-1, 0), bias=False)
        self.conv_bias = nn.Parameter(torch.zeros(1, self.out_c),requires_grad=True)

        # self.query_dim = self.out_c
        # self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.query_dim))
        # self.r_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)

        self.tanh = nn.Tanh()
        self.atten_sm = nn.Softmax()
        self.dropout = nn.Dropout(p=self.dropout_p, inplace=False)

        self.out_feature_size = self.out_c + self.features_size
        self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.out_feature_size))
        self.r_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)

        eye = torch.eye(self.out_feature_size, self.out_feature_size)
        self.att_W = nn.Parameter(eye.expand(self.n_rel, self.out_feature_size, self.out_feature_size), requires_grad=True)

        self.word_mem = Word_MEM(self.features_size, settings)
        self.rel_mem = Rel_MEM(self.out_feature_size, settings)

        self.pred_pos = nn.Parameter(torch.randn(self.n_rel, self.out_feature_size), requires_grad=True)
        self.pred_neg = nn.Parameter(torch.randn(self.n_rel, self.out_feature_size), requires_grad=True)
        self.pred_pos.data.uniform_(-0.01, 0.01)
        self.pred_neg.data.uniform_(-0.01, 0.01)
        self.pred_sm = nn.Sigmoid()
        self.pred_linear = nn.Linear(self.out_feature_size, 1)
        # self.pred_sm = nn.LogSoftmax()

        # con1 = math.sqrt(6.0 / ((self.pos_embed_size + self.word_embed_size)*self.window))
        con1 = 0.01
        nn.init.uniform_(self.conv.weight, a=-con1, b=con1)
        nn.init.uniform_(self.conv_bias, a=-con1, b=con1)

        # con = math.sqrt(6.0/(self.out_c + self.n_rel))
        con = 0.01
        nn.init.uniform_(self.r_embed, a=-con, b=con)
        nn.init.uniform_(self.r_bias, a=-con, b=con)

    def forward(self, inputs):
        bz = len(inputs)
        labels = [item['label'] for item in inputs]
        # list of bag of sentence reps, B * N_bag * D_sent
        sentence_embeds = self._create_sentence_embedding(inputs)
        s = self._fusion(sentence_embeds)  # B * n_rel * D_sent

        """
        # v1
        p_pos = torch.mul(s, self.pred_pos).sum(-1).unsqueeze(-1)  # B * n_rel
        p_neg = torch.mul(s, self.pred_neg).sum(-1).unsqueeze(-1)  # B * n_rel
        p = torch.cat([p_pos, p_neg], dim=-1)  # B * n_rel * 2
        p = self.dropout(p)
        pred = self.pred_sm(p)
        """

        """
        # v2
        if self.training:
            s = s[torch.arange(0, bz).long().cuda(), labels]
            # score is the same, but normalize over different set!
            scores = torch.matmul(s, self.r_embed.t()) + self.r_bias
            pred = self.pred_sm(scores)
        else:
            scores = torch.matmul(s, self.r_embed.t()) + self.r_bias
            pred = self.pred_sm(scores.view(-1, self.n_rel)).view(bz, self.n_rel, self.n_rel).max(1)[0]
        """
        # v3
        # same as in ﻿Jiang-16
        pred = self.dropout(self.pred_linear(s))
        # a dropout trick in Jiang-16 seems to have no effect.
        # pred = self.pred_sm(pred)
        # return probabilities
        return pred.view(-1, self.n_rel)

    def _create_sentence_embedding(self, inputs):
        batch_features = []
        # batch
        for ix in range(len(inputs)):
            bag = inputs[ix]['bag']
            features = []
            # bag of sentences
            for j, item in enumerate(bag):
                w2v = self.w2v(item.t()[0])

                pos1 = self.pos1_embed(item[:, 1])
                pos2 = self.pos2_embed(item[:, 2])
                word_feature = torch.cat([w2v, pos1, pos2], dim=-1).unsqueeze(0).unsqueeze(0)
                feature = self.conv(word_feature).squeeze(-1)
                feature = F.max_pool1d(feature, feature.size(-1)).squeeze(-1) + self.conv_bias
                # this tanh is little different from lin-16's.
                feature = self.tanh(feature)
                # feature = self.dropout(feature)

                # mem_feature (with position embedding)
                word_mem_feat = self.word_mem({
                    'w2v':word_feature.reshape(-1, self.features_size),
                    'sent':item,
                }).reshape(1, -1)  # 1 * feature_size

                feature = torch.cat([feature, word_mem_feat], dim=-1)  # 1 * (feature_size + out_c)
                features.append(feature)
            features = torch.cat(features, dim=0)
            batch_features.append(features)
        return batch_features

    def _fusion(self, sentence_embeds):
        '''
            input: list of bag of sentence reps, B * N_bag * D_sent
            return: bag_reps
        '''
        bag_reps = self._merge_instances(sentence_embeds)
        bag_reps = self.rel_mem(bag_reps)
        return bag_reps


    def _merge_instances(self, sentence_embeds):
        batch_features = []
        for features in sentence_embeds:
            atten_weights = self.atten_sm(torch.bmm(self.r_embed.unsqueeze(1),
                                                torch.matmul(self.att_W, features.t())).squeeze(1))
            features = torch.matmul(atten_weights, features)  # n_rel * out_feature_size
            '''
            if self.dropout is not None:
                features = self.dropout(features)
                if not self.training:
                    features = features * 0.5
            '''
            batch_features.append(features.unsqueeze(0))
        return torch.cat(batch_features, dim=0)




def split_query_and_memory(w2v, sent):
    pos1 = sent[0, 3]
    pos2 = sent[0, 4]
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1
    query = torch.cat([w2v[pos1], w2v[pos2]]).reshape(1, -1)  # 1 * 2d
    memory = torch.cat([w2v[:pos1], w2v[pos1+1:pos2], w2v[pos2+1:]]).reshape(-1, w2v.shape[-1])  # L_memory * d
    return query, memory


if __name__ == "__main__":
    root = "./origin_data/"
    train_data = Riedel_10(root)
    collate_fn = train_data.collate_fn
    train_loader = data.DataLoader(train_data,
                                        batch_size=32,
                                        pin_memory=False,
                                        shuffle=True,
                                        collate_fn=collate_fn)

    settings = {
        "out_feature_size" : 200,
        "memory_dim" : 100,
        "hop_size" : 3,
    }
    # model = Word_MEM(settings)
    # for item in train_loader:
    #     out = model(item)

