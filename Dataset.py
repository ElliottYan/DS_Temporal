import torch.utils.data as data

import os
import re
import torch
import pdb
import random
import pickle
# import attacker
from collections import defaultdict
import numpy as np
import torch
# from process import construct_dataset, create_labels
# from character_process import strToList, n_letters
from sklearn.preprocessing import normalize
import struct


class Riedel_10(data.Dataset):
    def __init__(self, root, train_test='train'):
        # filenames = ['train.txt', 'test.txt']
        self.root = root
        self.mode = train_test
        if self.mode == 'train':
            self.filename = 'train.txt'
        else:
            self.filename = 'test.txt'

        self.vec_name = 'vec.bin'
        self.rel_name = 'relation2id.txt'
        self.en_vec_path = "en_vecs/nyt_en_vecs.npy"
        self.en2id_path = "en_vecs/en2id.txt"

        self.read_in_vecs()
        self.en2id, self.en_vecs = self.read_in_en_vecs2()
        # relation position limit
        self.limit = 30
        self.prepare_data()
        self.keys = list(self.dict.keys())
        # self.n_rel = len(self.rel2id)
        # self.vocab_size = len(self.w2id)

    def read_in_vecs(self):
        self.w2v = []
        self.w2id = defaultdict(int)
        with open(os.path.join(self.root, self.vec_name), 'rb') as f:
            word_total, word_dimension = map(int, f.readline().strip().split())
            self.w2v.append(np.zeros((1, word_dimension)))
            self.w2id['UNK'] = 0
            for i in range(1, word_total+1):
                name = b''
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':
                        # pdb.set_trace()
                        # name += ch.decode('utf-8')
                        name += ch
                try:
                    name = name.decode('utf-8')
                except:
                    pdb.set_trace()
                self.w2id[name] = i
                vecs = struct.unpack('{}f'.format(word_dimension), f.read(word_dimension * 4))
                vecs = np.array(vecs).reshape(1, -1)
                vecs = normalize(vecs, norm='l2')
                self.w2v.append(vecs)
        word_total += 1
        self.vocab_size = word_total
        self.w2v = np.concatenate(self.w2v, axis=0)
        print('Finish reading in vecs!')
        # pdb.set_trace()

        self.rel2id = defaultdict(int)
        with open(os.path.join(self.root, self.rel_name), 'r') as f:
            lines = f.readlines()
        for line in lines:
            tmp = line.strip().split()
            rel = tmp[0]
            id = int(tmp[1])
            self.rel2id[rel] = id
        self.n_rel = len(lines)
        print('Finish reading in relation2id!')

    def prepare_data(self):
        self.dict = defaultdict(list)
        # pos1_max, pos1_min = 0, 0
        # pos2_max, pos2_min = 0, 0
        self.bag2rel = defaultdict(set)
        with open(os.path.join(self.root, self.filename), 'r') as f:
            lines = f.readlines()
        for line in lines:
            pos1, pos2 = 0, 0
            tmp = line.strip().split()
            id1, id2, e1, e2, rel = tmp[:5]
            rel = self.rel2id[rel]
            # sentence = list(map(lambda x: self.w2id[x], tmp[5:-1]))
            sentence = tmp[5:-1]
            sent = []
            for ix, word in enumerate(tmp[5:-1]):
                sent.append(self.w2id[word])
                if word == e1:
                    pos1 = ix
                if word == e2:
                    pos2 = ix

            sent_len = len(sentence)
            con = []
            for j in range(sent_len):
                conl = pos1 - j + self.limit
                conl = max(0, conl)
                conl = min(self.limit * 2, conl)
                conr = pos2 - j + self.limit
                conr = max(0, conr)
                conr = min(self.limit * 2, conr)
                con.append([sent[j], conl, conr])

            if self.mode == 'train':
                key = (e1, e2, rel)
            else:
                key = (e1, e2)
            self.dict[key].append(con)
            self.bag2rel[key].add(rel)

    def collate_fn(self, data):
        # pdb.set_trace()
        # return
        return data

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, item):
        key = self.keys[item]
        bag = []
        for item in self.dict[key]:
            bag.append(torch.cuda.LongTensor(item))

        # one-hot target
        # for rel in self.bag2rel[key]:
        #     target[rel] = 1
        # target = torch.
        target = self.bag2rel[key]
        target = torch.cuda.LongTensor(list(target))

        ret = {}
        ret['bag'] = bag
        ret['label'] = target
        ret['en_pair'] = (self.en2id[key[0]], self.en2id[key[1]])
        # return self.dict[key]
        return ret
    # def collate_fn(self):

    def read_in_en_vecs(self):
        vecs = np.load(self.en_vec_path)
        with open(self.en2id_path, 'r') as f:
            lines = f.readlines()
        en2id = {}
        for line in lines:
            key, name, val = line.strip().split('\t')
            en2id[key] = int(val)
        return en2id, vecs

    # in riedel's case, we only use the word embedding as entity-embedding for query construction
    def read_in_en_vecs2(self):
        with open(os.path.join(self.root, self.en2id_path), 'r') as f:
            lines = f.readlines()
        en2id = {}

        en_vecs = np.zeros((len(lines), 50))
        tmp_func = lambda x: self.w2v[self.w2id[x]]

        for line in lines:
            key, name, val = line.strip().split('\t')
            en2id[name] = int(val)
            en_vecs[int(val)] = tmp_func(name)
        return en2id, en_vecs


if __name__ == '__main__':
    root = '/data/yanjianhao/nlp/torch/torch_NRE/origin_data'
    dataset = Riedel_10(root)
