import torch
import torch.autograd as ag
import torch.nn as nn
import torch.optim as optim
import pdb
import torch.utils.data as data
import torch.nn.functional as F
import sklearn.metrics as metrics

import numpy as np
# from utils import pad_sequence, pad_labels, to_var, clip

torch.manual_seed(1)

class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num = 16, eps = 1e-10):
        super(GroupBatchnorm2d,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim = 2, keepdim = True)
        std = x.std(dim = 2, keepdim = True)

        x = (x - mean) / (std+self.eps)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta

# mem net model used for riedel model
class MEM_CNN_RIEDEL(nn.Module):
    def __init__(self, settings):
        super(MEM_CNN_RIEDEL, self).__init__()

        self.use_cuda = settings['use_cuda']
        self.cuda_devices = [4, 5, 6, 7]

        self.word_embed_size = settings['word_embed_size']
        self.vocab_size = settings['vocab_size']
        self.n_rel = settings['n_rel']
        self.hidden_size = 50
        self.w2v = nn.Embedding(self.vocab_size + 1, self.word_embed_size, padding_idx=self.vocab_size)
        self.features_size = settings['word_embed_size']

        self.pos_embed_size = 5
        # define the position embedding effective domain
        self.max_len = 60
        # number of output channels for CNN
        # self.out_c = 230
        self.out_c = settings['out_c']
        self.sent_feat_size = self.out_c
        self.dropout_p = settings['dropout_p']
        pre_word_embeds = settings['word_embeds']
        self.version = settings['version']

        self.position_embedding = settings['position_embedding']
        if settings['position_embedding']:
            self.features_size += 2 * self.pos_embed_size
            self.pos_embed = nn.Embedding(self.max_len * 2 + 1, self.pos_embed_size)

        # character embedding enabled
        character_embedding = False
        # if character_embedding:
        #     self.char_embed_size = 20
        #     self.timex_embed_size = self.char_embed_size
        #     self.char_embed = nn.Embedding(n_letters + 1, self.char_embed_size)
        #     self.features_size += self.char_embed_size
            # self.sent_feat_size += self.timex_embed_size

        # entity embeddings for creating queries
        self.en_embed_size = settings['entity_embed_size']
        # self.n_entity = settings['n_entity']
        pre_en_embeds = settings['entity_pretrained_vecs']
        if pre_en_embeds is not None:
            self.en_embed = nn.Embedding(*pre_en_embeds.shape)
            # here pre_en_embeds are np arrays
            # pdb.set_trace()
            pre_length = pre_en_embeds.shape[0]
            self.en_embed.weight.data[:pre_length] = torch.FloatTensor(pre_en_embeds)

        self.bag_size = 30
        self.out_feature_size = self.out_c

        order_embed = None
        # todo : somehow, order embed should be consistent with time orders
        if order_embed is not None:
            self.order_embed_size = 50
            self.order_embed = nn.Embedding(self.bag_size, self.order_embed_size, padding_idx=self.bag_size-1)
            self.out_feature_size = self.out_c + self.order_embed_size

        # word embedding
        if pre_word_embeds is not None:
            self.pre_word_embed = True
            self.word_embed.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds), requires_grad=True)

        else:
            self.pre_word_embed = False

        self.query_dim = 100
        self.phi_q = nn.Parameter(torch.randn(self.en_embed_size, self.query_dim), requires_grad=True)

        self.atten_sm = nn.Softmax()

        # todo : here we only have one convolution layer, the results only depends on 3 words windows.
        # cannot case big picture results
        # maybe more layers of CNN
        self.window = 3
        self.conv = nn.Conv2d(1, self.out_c, (self.window, self.features_size), padding=(self.window-1, 0))
        # self.conv4 = nn.Conv2d(1, self.out_c, (4, self.features_size))
        # self.conv5 = nn.Conv2d(1, self.out_c, (5, self.features_size))

        # normalization over CNN outputs
        self.group_norm = GroupBatchnorm2d(self.out_c)

        self.max_hops = settings['max_hops']
        memory_dim = self.query_dim
        for i in range(self.max_hops):
            # also add bias ?
            C = nn.Linear(self.out_feature_size, memory_dim, bias=False)
            C.weight.data.normal_(0, 0.1)
            self.add_module('C_{}'.format(i), C)
        self.C = AttrProxy(self, 'C_')

        # relation embedding size is the same as MEM's output
        self.r_embed = nn.Embedding(self.n_rel, self.query_dim)
        self.r_bias = nn.Parameter(torch.randn(self.n_rel), requires_grad=True)
        self.bilinear = nn.Parameter(torch.randn(memory_dim, self.query_dim), requires_grad=True)

        self.linear = nn.Linear(memory_dim, self.n_rel)
        # NLL loss is apllied to logit outputs
        self.M_embed = nn.Parameter(torch.rand(self.n_rel, memory_dim), requires_grad=True)

        eye = torch.eye(self.out_c, self.out_c)
        self.att_W = nn.Parameter(eye.expand(self.n_rel, self.out_c, self.out_c), requires_grad=True)

        self.pred_sm = nn.LogSoftmax()
        # only use for debugging
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=self.dropout_p, inplace=True)

    # position embedding for each word in sentence
    # def _create_position_embed(self, sent_len, pos1, pos2):
        # above = torch.Tensor([self.max_len]).expand(sent_len)
        # below = torch.Tensor([-1 * self.max_len]).expand(sent_len)
        # lookup tensor should be all positive
        # pf1_lookup = ag.Variable(clip(torch.arange(0, sent_len) - float(pos1),
        #                               above, below).cuda().long() + self.max_len)
        # pf2_lookup = ag.Variable(clip(torch.arange(0, sent_len) - float(pos2),
        #                               above, below).cuda().long() + self.max_len)
        # pf1 = self.pos_embed(pf1_lookup)
        # pf2 = self.pos_embed(pf2_lookup)

        # return torch.cat([pf1, pf2], dim=1)

    def forward(self, inputs):
        # the return of cnn is stored as memories for latter query.
        mem_bags = self._create_sentence_embedding(inputs)
        # first I try the queries with time.
        # queries = self._create_queries_2(kwargs['en_pairs'])
        # queries = self._create_queries_3(inputs)
        queries = self._create_queries_2(inputs)
        labels = [item['label'] for item in inputs]
        predicts = self._predict_bag(mem_bags, queries, labels=labels)
        # predict = self.pred_sm()
        return predicts

    def _create_sentence_embedding(self, inputs):
        bags = [item['bag'] for item in inputs]
        batch_features = []
        for ix, bag in enumerate(bags):
            # pdb.set_trace()
            features = []
            for item in bag:
                w2v = self.w2v(item.t()[0])
                # this may need some modification for further use.
                pos1 = self.pos1_embed(item[:, 1])
                pos2 = self.pos1_embed(item[:, 2])
                feature = torch.cat([w2v, pos1, pos2], dim=-1).unsqueeze(0).unsqueeze(0)
                feature = self.conv(feature).squeeze(-1)
                feature = F.max_pool1d(feature, feature.size(-1)).squeeze(-1) + self.conv_bias
                # this tanh is little different from lin-16's.
                feature = self.tanh(feature)
                feature = self.dropout(feature)
                # dropout is a little different too.
                features.append(feature)
            features = torch.cat(features, dim=0)

            atten_weights = self.atten_sm(torch.bmm(self.r_embed.unsqueeze(1),
                                                    torch.matmul(self.att_W, features.t())).squeeze(1))
            # to this point, features is actually s
            # n_rel * D
            features = torch.matmul(atten_weights, features)
            if self.dropout is not None:
                # pdb.set_trace()
                features = self.dropout(features)
                # pdb.set_trace()
                if not self.training:
                    features = features * 0.5

            batch_features.append(features.unsqueeze(0))
        return torch.cat(batch_features, dim=0)

    def _create_queries(self, en_pairs):
        lookup_var = torch.LongTensor(en_pairs)
        batch_en_embeds = self.en_embed(lookup_var)
        en1_embeds, en2_embeds = torch.split(batch_en_embeds, 1, dim=1)
        # one ways to compute queries
        queries = torch.matmul((en1_embeds + en2_embeds).view(-1, self.en_embed_size), self.phi_q)

        return queries

    def _create_queries_2(self, inputs):
        en_pairs = [item['en_pair'] for item in inputs]
        batch_size = len(en_pairs)
        lookup_var = torch.LongTensor(en_pairs, requires_grad=True)
        batch_en_embeds = self.en_embed(lookup_var)
        en1_embeds, en2_embeds = torch.split(batch_en_embeds, 1, dim=1)
        entity_queries = torch.matmul((en1_embeds + en2_embeds).view(-1, self.en_embed_size), self.phi_q)
        queries = self.r_embed.weight.expand((batch_size, ) + self.r_embed.weight.size()) + entity_queries.unsqueeze(1)
        return queries

    def _create_queries_3(self, inputs):
        batch_size = len(inputs)
        # without entity queries
        queries = self.r_embed.weight.expand((batch_size, ) + self.r_embed.weight.size())
        return queries

    # In this version, I use standard memN2N structure
    def _predict(self, mem_bags, queries, labels=None):
        ret = []
        # trained one by one
        # memory is bag_size * out_dim
        for ix, memory in enumerate(mem_bags):
            # query here is with size : n_rel * D
            query_r = queries[ix]
            query_r = query_r.expand((memory.size(0),) + query_r.size())

            # maybe should consider limit the bag size of inputs
            # todo : need more serious thoughts
            lookup_tensor = torch.LongTensor(
                list(range(memory.size(0)))[:self.bag_size] + \
                    (memory.size(0) - self.bag_size) * [self.bag_size - 1],
                requires_grad=True)
            # lookup_var = to_var(lookup_tensor)
            # order_embed : bag_size * order_embed_size
            order_embed = self.order_embed(lookup_tensor)
            # query : bag_size * n_rel * (order_embed_size + query_size)
            # each indicates a query for answer of one relation
            query = torch.cat([query_r, order_embed.view(order_embed.size(0), 1, order_embed.size(-1)).expand(order_embed.size(0), self.n_rel, order_embed.size(-1))], dim=-1)
            memory = torch.cat([memory, order_embed], dim=-1)
            # query_embed = self.C[0](query)
            for hop in range(self.max_hops):
                # todo : maybe a key-val embedding
                m_key = self.C[0](memory)
                m_val = self.C[1](memory)

                # softmax need 2D tensor
                # each query over all memories
                tmp = torch.matmul(query, m_key.t())
                prob_size = tmp.size()
                prob = self.atten_sm(tmp.view(-1, m_key.size(0)))
                prob = prob.view(prob_size)

                # for each query and each relation
                o_k = torch.matmul(prob, m_val)

                # update its query_r
                query = query + o_k
            # todo: use attention cnn ways to embed relation.
            # can substract the query vector out to find which is our target.
            # query = query - self.r_embed.weight
            query = self.dropout(query)
            # change the dimension for output query
            # bi-linear form
            query = torch.matmul(query, self.bilinear)
        # if self.training:

        # else:
            tmp = torch.matmul(query, self.r_embed.weight.t())
            scores = []
            for i in range(tmp.size(0)):
                scores.append(tmp[i].diag() + self.r_bias)
            ret.append(self.pred_sm(torch.stack(scores)))
        return ret

    # this is for riedel-10 dataset
    def _predict_bag(self, mem_bags, queries, labels=None):
        if labels is not None:
            labels = labels.cuda()
        ret = []
        # trained one by one
        # memory is bag_size * out_dim
        for ix, memory in enumerate(mem_bags):
            # query here is with size : n_rel * D
            query_r = queries[ix]
            query_r = query_r.expand((1,) + query_r.size())

            # query : bag_size * n_rel * query_size
            query = query_r

            for hop in range(self.max_hops):
                # key val version
                if self.version == 1:
                    m_key = self.C[0](memory)
                    m_val = self.C[1](memory)
                # layer sharing version
                else:
                    m_key = self.C[hop](memory)
                    m_val = self.C[hop](memory)
                # softmax need 2D tensor
                # each query over all memories
                tmp = torch.matmul(query, m_key.t())
                prob_size = tmp.size()
                prob = self.atten_sm(tmp.view(-1, m_key.size(0)))
                prob = prob.view(prob_size)

                # for each query and each relation
                o_k = torch.matmul(prob, m_val)

                # update its query_r
                query = query + o_k
            # todo: use attention cnn ways to embed relation.
            # can substract the query vector out to find which is our target.
            # query = query - self.r_embed.weight
            query = self.dropout(query)

            # testing this schema
            modified = False
            if modified:
                # labels = labels[ix]
                if self.training:
                    # pdb.set_trace()
                    query = query[0][labels[ix]]
                    # score is the same, but normalize over different set!
                    scores = torch.matmul(query, self.r_embed.weight.t()) + self.r_bias
                    # pdb.set_trace()
                    # scores = self.pred_sm(scores.view(1, -1))
                else:
                    scores = torch.matmul(query, self.r_embed.weight.t()) + self.r_bias
                    scores = self.pred_sm(scores.view(-1, self.n_rel)).view(bz, self.n_rel, self.n_rel).max(1)[0]

            else:
                scores = self.pred_sm((query * self.r_embed.weight).sum(dim=-1) + self.r_bias)
            ret.append(scores)
        return ret
