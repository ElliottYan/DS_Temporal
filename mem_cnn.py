import torch
import torch.autograd as ag
import torch.nn as nn
import torch.optim as optim
import pdb
import torch.utils.data as data
import torch.nn.functional as F
import sklearn.metrics as metrics
import math
import numpy as np
# from utils import pad_sequence, pad_labels, to_var, clip

torch.cuda.manual_seed(1)
torch.manual_seed(1)

class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

# the position embedding of attention is all you need.
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
def position_encoding_init(n_position, d_pos_vector):
    position_encode = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vector) for j in range(d_pos_vector)]
        for pos in range(n_position)
    ])
    position_encode[:, 0::2] = np.sin(position_encode[:, 0::2])
    position_encode[:, 1::2] = np.cos(position_encode[:, 1::2])
    return torch.from_numpy(position_encode).type(torch.FloatTensor)


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
        self.features_size = settings['word_embed_size']
        self.tri_attention = settings['tri_attention']


        self.pos_embed_size = 5
        self.features_size += 2 * self.pos_embed_size
        # define the position embedding effective domain
        # self.max_len = 60
        self.pos_limit = settings['pos_limit']
        # number of output channels for CNN
        # self.out_c = 230
        self.out_c = settings['out_c']
        self.sent_feat_size = self.out_c
        self.dropout_p = settings['dropout_p']
        pre_word_embeds = settings['word_embeds']
        self.version = settings['version']
        self.remove_origin_query = settings['remove_origin_query']

        self.w2v = nn.Embedding(self.vocab_size, self.word_embed_size, padding_idx=self.vocab_size-1)
        # word embedding
        if pre_word_embeds is not None:
            self.pre_word_embed = True
            self.w2v.weight.data[:pre_word_embeds.shape[0]] = nn.Parameter(torch.FloatTensor(pre_word_embeds), requires_grad=True)
        else:
            self.pre_word_embed = False

        self.position_embedding = settings['position_embedding']
        if settings['position_embedding']:
            self.pos1_embed = nn.Embedding(self.pos_limit * 2 + 1, self.pos_embed_size)
            self.pos2_embed = nn.Embedding(self.pos_limit * 2 + 1, self.pos_embed_size)
        else:
            self.features_size -= 2 * self.pos_embed_size

        # too : here we only have one convolution layer, the results only depends on 3 words windows.
        # cannot case big picture results
        # maybe more layers of CNN
        self.window = 3
        self.conv = nn.Conv2d(1, self.out_c, (self.window, self.features_size), padding=(self.window-1, 0), bias=False)
        self.conv_bias = nn.Parameter(torch.zeros(1, self.out_c),requires_grad=True)

        self.memory_decay_weight = settings['memory_decay_weight']

        # entity embeddings for creating queries
        self.en_embed_size = settings['entity_embed_size']
        # self.n_entity = settings['n_entity']
        pre_en_embeds = settings['entity_pretrained_vecs']
        if pre_en_embeds is not None:
            self.en_embed = nn.Embedding(*pre_en_embeds.shape)
            # here pre_en_embeds are np arrays
            pre_length = pre_en_embeds.shape[0]
            self.en_embed.weight.data[:pre_length] = torch.FloatTensor(pre_en_embeds)

        self.bag_size = 30
        self.out_feature_size = self.out_c

        self.bag_size = 30
        order_embed = settings['order_embed']
        if order_embed is True:
            self.order_embed_size = 50
            self.order_embed = nn.Embedding(self.bag_size, self.order_embed_size, padding_idx=0)
            self.out_feature_size = self.out_feature_size + self.order_embed_size
            tmp_con = math.sqrt(6.0 / (self.bag_size + self.order_embed_size))
            nn.init.uniform_(self.order_embed.weight, a=-tmp_con, b=tmp_con)
            circular_embedding = settings['circular']
            if circular_embedding:
                self.order_embed.weight.data = position_encoding_init(self.bag_size, self.order_embed_size)
                self.order_embed.weight.requires_grad = False
        else:
            self.order_embed = None

        self.use_rank = False

        if not settings['scalable_circular']:
            self.order_weight = settings['order_weight']
        else:
            self.order_weight = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.out_c), requires_grad=True)
        self.r_bias = nn.Parameter(torch.zeros(self.n_rel), requires_grad=True)

        self.ln = nn.LayerNorm(self.out_feature_size)

        self.query_dim = self.out_feature_size
        self.phi_q = nn.Parameter(torch.randn(self.en_embed_size, self.query_dim), requires_grad=True)

        self.atten_sm = nn.Softmax(dim=-1)
        self.pred_sm = nn.LogSoftmax(dim=-1)


        # normalization over CNN outputs
        # self.group_norm = GroupBatchnorm2d(self.out_c)

        self.max_hops = settings['max_hops']
        self.hop_size = 2 if self.version else self.max_hops
        memory_dim = self.query_dim
        for i in range(self.hop_size):
            # also add bias ?
            C = nn.Linear(self.out_feature_size, memory_dim, bias=False)
            # C.weight.data.normal_(0, 0.1)
            C.weight.data = torch.diag(torch.ones(memory_dim))
            self.add_module('C_{}'.format(i), C)
        self.C = AttrProxy(self, 'C_')

        self.query_type = settings['query_type']
        if self.query_type == 'SELF':
            self.M = nn.Linear(self.out_feature_size, memory_dim, bias=False)

        # relation embedding size is the same as MEM's output
        eye = torch.eye(self.out_c, self.out_c)
        self.att_W = nn.Parameter(eye.expand(self.n_rel, self.out_c, self.out_c), requires_grad=True)

        # NLL loss is apllied to logit outputs

        # only use for debugging
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=self.dropout_p, inplace=False)

        con1 = math.sqrt(6.0 / ((self.pos_embed_size + self.word_embed_size)*self.window))
        nn.init.uniform_(self.conv.weight, a=-con1, b=con1)
        nn.init.uniform_(self.conv_bias, a=-con1, b=con1)

        con = math.sqrt(6.0/(self.out_c + self.n_rel))
        nn.init.uniform_(self.r_embed, a=-con, b=con)
        nn.init.uniform_(self.r_bias, a=-con, b=con)

    def forward(self, inputs):
        # the return of cnn is stored as memories for latter query.
        bz = len(inputs)
        # pdb.set_trace()
        mem_bags = self._create_sentence_embedding(inputs)
        # first I try the queries with time.
        # queries = self._create_queries_2(kwargs['en_pairs'])
        if self.query_type == 'ENTITY':
            queries = self._create_queries_2(inputs)
        elif self.query_type == 'SELF':
            queries = self._create_queries_4(inputs, encoding_output=mem_bags)
        else:
            queries = self._create_queries_3(inputs)

        # queries = self._create_queries_2(inputs)
        labels = [item['label'] for item in inputs]
        predicts = self._predict_bag(mem_bags, queries, labels=labels)
        if self.training:
            predicts = predicts[torch.arange(0, bz).long().cuda(), labels]
            # score is the same, but normalize over different set!
            scores = torch.matmul(predicts, self.r_embed.t()) + self.r_bias
            pred = self.pred_sm(scores)
        else:
            scores = torch.matmul(predicts, self.r_embed.t()) + self.r_bias
            pred = self.pred_sm(scores.view(-1, self.n_rel)).view(bz, self.n_rel, self.n_rel).max(1)[0]
        return pred

    # todo: also wanna use self-attention to form the encoding part.
    def _create_sentence_embedding(self, inputs):
        bags = [item['bag'] for item in inputs]
        batch_features = []
        for ix, bag in enumerate(bags):
            features = []
            for item in bag:
                w2v = self.w2v(item.t()[0])
                # this may need some modification for further use.
                pos1 = self.pos1_embed(item[:, 1])
                pos2 = self.pos2_embed(item[:, 2])
                feature = torch.cat([w2v, pos1, pos2], dim=-1).unsqueeze(0).unsqueeze(0)
                feature = self.conv(feature).squeeze(-1)
                feature = F.max_pool1d(feature, feature.size(-1)).squeeze(-1) + self.conv_bias
                # this tanh is little different from lin-16's.
                feature = self.tanh(feature)
                feature = self.dropout(feature)
                # dropout is a little different too.
                features.append(feature)

            # shape : bag_size * D
            features = torch.cat(features, dim=0)
            features = self.dropout(features)
            # features = self.ln(features)

            batch_features.append(features)
        return batch_features

    def _create_queries(self, en_pairs):
        lookup_var = torch.cuda.LongTensor(en_pairs)
        batch_en_embeds = self.en_embed(lookup_var)
        en1_embeds, en2_embeds = torch.split(batch_en_embeds, 1, dim=1)
        # one ways to compute queries
        queries = torch.matmul((en1_embeds + en2_embeds).view(-1, self.en_embed_size), self.phi_q)

        return queries

    # query_type : entity + r_embed
    def _create_queries_2(self, inputs, encoding_output=None):
        en_pairs = [item['en_pair'] for item in inputs]
        batch_size = len(en_pairs)
        lookup_var = torch.cuda.LongTensor(en_pairs)
        batch_en_embeds = self.en_embed(lookup_var)
        en1_embeds, en2_embeds = torch.split(batch_en_embeds, 1, dim=1)
        entity_queries = torch.matmul((en1_embeds + en2_embeds).view(-1, self.en_embed_size), self.phi_q)
        # queries = self.r_embed.expand((batch_size, ) + self.r_embed.size()) + entity_queries.unsqueeze(1)
        queries = torch.stack([self.r_embed] * batch_size) + entity_queries.unsqueeze(1)
        return queries

    # query_type : r_embed
    def _create_queries_3(self, inputs, encoding_output=None):
        batch_size = len(inputs)
        # without entity queries
        # queries = self.r_embed.expand((batch_size, ) + self.r_embed.size())
        # queries = []
        # for _ in range(batch_size):
        #     queries.append(self.r_embed)
        queries = torch.stack([self.r_embed] * batch_size)
        return queries

    # query_type : self
    def _create_queries_4(self, inputs, encoding_output=None):
        bz = len(inputs)
        ret = []
        for item in encoding_output:
            ret.append(self.M(item))
        return ret


    # this is for riedel-10 dataset
    def _predict_bag(self, mem_bags, queries, labels=None):
        bz = len(labels)
        ret = []
        # trained one by one
        # memory is bag_size * out_dim
        for ix, memory in enumerate(mem_bags):
            # query here is with size : n_rel * D
            query_r = queries[ix]

            # query : bag_size * n_rel * query_size
            query = query_r

            for hop in range(self.max_hops):
                # key val version
                if self.version == 1:
                    m_key = self.C[0](memory)
                    m_val = self.C[1](memory)
                # layer sharing version
                elif self.version == 0:
                    m_key = self.C[hop](memory)
                    m_val = self.C[hop](memory)
                else:
                    m_key = memory
                    m_val = memory
                # softmax need 2D tensor
                # each query over all memories
                # tmp = torch.matmul(query, m_key.t())
                if self.query_type != 'SELF':
                    # tmp = torch.bmm(query.unsqueeze(1),
                    #                 torch.matmul(self.att_W, m_key.t())).squeeze(1)
                    tmp = torch.matmul(query, m_key.t())
                else:
                    tmp = torch.matmul(query, m_key.t())

                prob_size = tmp.size()
                prob = self.atten_sm(tmp.view(-1, m_key.size(0)))
                prob = prob.view(prob_size)

                # for each query and each relation
                o_k = torch.matmul(prob, m_val)

                # update its query_r

                # query = query + o_k * self.memory_decay_weight
                query = query + o_k

            # can substract the query vector out to find which is our target.
            # only when D_r == D_q
            # can be compatible with different construction of queries.
            if self.remove_origin_query:
                query = query - query_r

            # additional selective attention is applied
            if self.query_type == 'SELF':
                atten_weights = self.atten_sm(torch.bmm(self.r_embed.unsqueeze(1),
                                                        torch.matmul(self.att_W, query.t())).squeeze(1))
                # n_rel * D
                query = torch.matmul(atten_weights, query)

            if self.dropout is not None:
                query = self.dropout(query)
                if not self.training:
                    query = query * 0.5

            # testing this scheme
            # modified = False
            # if modified:
            #     labels = labels[ix]
                # if self.training:
                #     query = query[0][labels[ix]]
                #     score is the same, but normalize over different set!
                #     scores = torch.matmul(query, self.r_embed.t()) + self.r_bias
                #     scores = self.pred_sm(scores.view(1, -1))
                # else:
                #     scores = torch.matmul(query, self.r_embed.t()) + self.r_bias
                #     scores = self.pred_sm(scores.view(-1, self.n_rel)).view(bz, self.n_rel, self.n_rel).max(1)[0]
            #
            # else:
            #     scores = self.pred_sm((query * self.r_embed).sum(dim=-1) + self.r_bias)
            ret.append(query)
            # ret.append(scores)
        ans = torch.stack(ret)
        return ans

class MEM_CNN_WIKI(MEM_CNN_RIEDEL):
    def __init__(self, settings):
        super(MEM_CNN_WIKI, self).__init__(settings)
        self.bilinear = nn.Parameter(torch.randn(self.query_dim, self.out_c), requires_grad=True)

        self.query_dim = 100
        memory_dim = self.query_dim
        if self.order_embed is not None:
            memory_dim += self.order_embed_size
        for i in range(self.hop_size):
            # also add bias ?
            C = nn.Linear(self.out_feature_size, memory_dim, bias=False)
            C.weight.data.normal_(0, 0.1)
            self.add_module('C_{}'.format(i), C)
        self.C = AttrProxy(self, 'C_')

        self.use_rank = settings['use_rank']

        # relation embedding size is the same as MEM's output
        self.r_embed = nn.Parameter(torch.zeros(self.n_rel, self.query_dim))
        self.r_bias = nn.Parameter(torch.randn(self.n_rel), requires_grad=True)
        con = math.sqrt(6.0/(self.query_dim + self.n_rel))
        nn.init.uniform_(self.r_embed, a=-con, b=con)
        nn.init.uniform_(self.r_bias, a=-con, b=con)

        self.query_last = settings['query_last']

        self.bilinear = nn.Parameter(torch.randn(memory_dim, self.query_dim), requires_grad=True)

        self.M_embed = nn.Parameter(torch.rand(self.n_rel, memory_dim), requires_grad=True)

    def forward(self, inputs):
        # the return of cnn is stored as memories for latter query.
        bz = len(inputs)
        # pdb.set_trace()
        mem_bags = self._create_sentence_embedding(inputs)
        # first I try the queries with time.
        # queries = self._create_queries_2(kwargs['en_pairs'])
        if self.query_type == 'ENTITY':
            queries = self._create_queries_2(inputs)
        elif self.query_type == 'SELF':
            queries = self._create_queries_4(inputs, encoding_output=mem_bags)
        else:
            queries = self._create_queries_3(inputs)

        # queries = self._create_queries_2(inputs)
        labels = [item['label'] for item in inputs]
        pred = self._predict(mem_bags, queries, labels=labels, inputs=inputs)
        return pred

    # predcit function for entity-advised query
    def _predict_with_entity(self, mem_bags, queries, labels=None):
        ret = []
        # trained one by one
        # memory is bag_size * out_dim
        for ix, memory in enumerate(mem_bags):
            # query here is with size : 1 * query_dim
            query_r = queries[ix]
            if not self.query_last:
                query_r = torch.cat([query_r] * memory.size(0), dim=0)
            else:
                query_r = query_r.reshape(1, -1)

            # maybe should consider limit the bag size of inputs
            # todo : need more serious thoughts
            lookup_tensor = torch.LongTensor(
                list(range(memory.size(0)))[-self.bag_size:] + \
                (memory.size(0) - self.bag_size) * [0], requires_grad=True)
            # order_embed : bag_size * order_embed_size
            order_embed = self.order_embed(lookup_tensor)
            order_embed *= self.order_weight
            # query : bag_size * 1 * (order_embed_size + query_size)
            # each indicates a query for answer of one relation
            query = torch.cat([query_r, order_embed], dim=-1)
            memory = torch.cat([memory, order_embed], dim=-1)
            # query_embed = self.C[0](query)

            # version is denoting key-val or layer-wise
            # version = 2
            for hop in range(self.max_hops):
                #  maybe a key-val embedding, maybe layer embedding.
                if self.version == 1:
                    m_key = self.C[0](memory)
                    m_val = self.C[1](memory)
                else:
                    m_key = self.C[hop](memory)
                    m_val = self.C[hop](memory)

                # softmax need 2D tensor
                # each query over all memories
                tmp = torch.matmul(query, m_key.t())
                # adding attention-bias
                prob = self.atten_sm(tmp)
                # prob = prob.view(prob_size)

                # for each query and each relation
                o_k = torch.matmul(prob, m_val)

                # update its query_r
                query = query + o_k
            # todo: use attention cnn ways to embed relation.
            # can substract the query vector out to find which is our target.
            # query -= query_r
            if self.debug:
                self.debug = False
            query = self.dropout(query)
            scores = torch.matmul(query, self.bilinear)

            ret.append(self.pred_sm(scores))
        return ret

    # In this version, I use standard memN2N structure
    def _predict(self, mem_bags, queries, labels=None, inputs=None):
        ret = []
        # trained one by one
        # memory is bag_size * out_dim
        for ix, memory in enumerate(mem_bags):
            # query here is with size : n_rel * D
            query_r = queries[ix]
            if not self.query_last:
                # query_r = torch.cat([query_r] * memory.size(0), dim=0)
                query_r = torch.stack([query_r] * memory.size(0))
            else:
                query_r = query_r.unsqueeze(0)
            query = query_r

            # maybe should consider limit the bag size of inputs
            # todo : need more serious thoughts
            if self.use_rank:
                # padding at first
                lookup_tensor = torch.cuda.LongTensor(inputs[ix]['ranks'])
                lookup_tensor += self.bag_size - lookup_tensor[-1] - 1
                lookup_tensor = lookup_tensor.clamp(0, self.bag_size-1)
            else:
                lookup_tensor = torch.cuda.LongTensor(
                    list(range(memory.size(0)))[:self.bag_size] + \
                    (memory.size(0) - self.bag_size) * [0])
            # lookup_tensor = torch.tensor(lookup_tensor, =True)

            # order_embed : bag_size * order_embed_size
            if self.order_embed is not None:
                order_embed = self.order_weight * self.order_embed(lookup_tensor)
                memory = torch.cat([memory, order_embed], dim=-1)
                order_embed = order_embed.view(order_embed.size(0), 1, order_embed.size(-1))
                order_embed = torch.cat([order_embed] * self.n_rel, dim=1)
                # query : query_size (maybe bag_size) * n_rel * (order_embed_size + query_size)

                # suit the case when memory_size != query_size
                order_embed = order_embed[:query.size(0)]
                # each indicates a query for answer of one relation
                query = torch.cat([query_r, order_embed], dim=-1)

            attention_bias = torch.triu(torch.ones(query.size(0), query.size(0)), diagonal=1) * (-1e9)
            attention_bias = attention_bias.cuda()

            # version is denoting key-val or layer-wise
            # version = 2
            for hop in range(self.max_hops):
                #  maybe a key-val embedding, maybe layer embedding.
                if self.version == 1:
                    m_key = self.C[0](memory)
                    m_val = self.C[1](memory)
                else:
                    m_key = self.C[hop](memory)
                    m_val = self.C[hop](memory)

                # softmax need 2D tensor
                # each query over all memories
                tmp = torch.matmul(query, m_key.t())
                if self.tri_attention:
                    tmp += attention_bias.unsqueeze(1) + tmp

                # adding attention_bias

                prob_size = tmp.size()
                prob = self.atten_sm(tmp.view(-1, m_key.size(0)))
                prob = prob.view(prob_size)

                # for each query and each relation
                o_k = torch.matmul(prob, m_val)

                # update its query_r
                try:
                    query = query + o_k
                except:
                    pdb.set_trace()
            # can substract the query vector out to find which is our target.
            query = self.dropout(query)
            if self.order_embed is not None:
                query = torch.matmul(query, self.bilinear)

            # after changing with attention pcnn
            modified = False
            if modified:
                # query size is same as memory size
                if self.training:
                    query = query[torch.arange(0, query.size(0)).cuda().long(), labels[ix].cuda()]
                    # score is the same, but normalize over different set!

                    scores = torch.matmul(query, self.r_embed.t()) + self.r_bias
                    scores = self.pred_sm(scores.view(memory.size(0), -1))
                    # pdb.set_trace()
                    # scores = self.pred_sm(scores.view(1, -1))
                else:
                    scores = torch.matmul(query, self.r_embed.t()) + self.r_bias
                    scores = self.pred_sm(scores.view(-1, self.n_rel)).view(scores.size(0), self.n_rel, self.n_rel).max(1)[0].view(scores.size(0), -1)
            else:
                scores = (query * self.r_embed).sum(dim=-1) + self.r_bias
                scores = self.pred_sm(scores.view(query.size(0), -1))

            ret.append(self.pred_sm(scores))
        return ret