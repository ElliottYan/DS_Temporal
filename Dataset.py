import torch.utils.data as data

import os
import re
import torch
import pdb
import random
import pickle
import process
# import attacker
from collections import defaultdict, Counter
from itertools import chain
import numpy as np
import struct
import torch
from sklearn.preprocessing import normalize

random.seed(10)

import logging
logging.basicConfig(level=logging.INFO)

class NYT_10(data.Dataset):
    def __init__(self, root, train_test='train', debug=False, use_whole_bag=True, **kwargs):
        # filenames = ['train.txt', 'test.txt']
        self.root = root
        self.mode = train_test
        self.use_whole_bag = use_whole_bag
        end_modifier = '.txt'
        if debug:
            print('In debug mode !')
            end_modifier = '.head'
        self.filename = train_test + end_modifier

        self.vec_name = 'vec.bin'
        self.rel_name = 'relation2id.txt'
        self.en_vec_path = "en_vecs/nyt_en_vecs.npy"
        self.en2id_path = "en_vecs/en2id.txt"

        self.read_in_vecs()
        self.en2id, self.en_vecs = self.read_in_en_vecs2()
        # relation position limit
        self.limit = 30
        self.max_sent_len = 100
        self.max_bag_size = 200
        self.prepare_data()
        self.keys = sorted(list(self.dict.keys()))
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
        self.pos_idx = dict()
        with open(os.path.join(self.root, self.filename), 'r', encoding='utf8') as f:
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

            # filter the sentence that are too long.
            if len(sent) > self.max_sent_len and self.mode == 'train':
                continue

            sent_len = len(sentence)
            con = []
            for j in range(sent_len):
                conl = pos1 - j + self.limit
                conl = max(0, conl)
                conl = min(self.limit * 2, conl)
                conr = pos2 - j + self.limit
                conr = max(0, conr)
                conr = min(self.limit * 2, conr)
                con.append([sent[j], conl, conr, pos1, pos2])

            # use_whole_bag = True
            use_whole_bag = self.use_whole_bag
            if self.mode == 'train' and not use_whole_bag:
                key = (e1, e2, rel)
            else:
                key = (e1, e2)
            self.pos_idx[key] = (pos1, pos2)
            self.dict[key].append(con)
            self.bag2rel[key].add(rel)

        # collect the data counter for each key item.
        # with open(os.path.join(self.root, 'tmp_' + self.mode + '.txt'), 'w') as f:
        #     for key, value in self.dict.items():
        #         f.write(str(key) + ':' + str(len(value)) + '\n')
        #
        # with open(os.path.join(self.root, 'tmp_sent_' + self.mode + '.txt'), 'w') as f:
        #     c = Counter()
        #     c.update([len(it) for it in chain(self.dict.values())])
        #     for key, value in c.items():
        #         f.write(str(key) + ':' + str(value) + '\n')

        # only filter the data in training.
        if self.mode == 'train':
            self.filter_bag_size()

    # filter bags that are too big.
    def filter_bag_size(self):
        pop_keys = []
        for key in self.dict.keys():
            dict_val = self.dict[key]
            if len(dict_val) > self.max_bag_size:
                pop_keys.append(key)
        # dict size cannot be changed during iteration.
        for key in pop_keys:
            self.dict.pop(key)
            self.pos_idx.pop(key)
            self.bag2rel.pop(key)

    def collate_fn(self, data):
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
        # target in shape [1]
        target = torch.cuda.LongTensor(list(target))

        ret = {}
        ret['bag'] = bag
        ret['label'] = target
        ret['en_pair'] = (self.en2id[key[0]], self.en2id[key[1]])
        # ret['pos1_idx'] = self.pos_idx[key][0]
        # ret['pos2_idx'] = self.pos_idx[key][1]
        # return self.dict[key]
        return ret

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
        with open(os.path.join(self.root, self.en2id_path), 'r', encoding='utf8') as f:
            lines = f.readlines()
        en2id = {}

        en_vecs = np.zeros((len(lines), 50))
        tmp_func = lambda x: self.w2v[self.w2id[x]]

        for line in lines:
            key, name, val = line.strip().split('\t')
            en2id[name] = int(val)
            en_vecs[int(val)] = tmp_func(name)
        return en2id, en_vecs


class WIKI_TIME(data.Dataset):
    def __init__(self, root, train_test='train', transform=None, position_embed=True, debug=False, construct=False, **kwargs):
        # construct file name
        if train_test == 'train':
            file_name = 'mini_train_temporal_v2.txt'
        elif train_test == "test":
            file_name = 'mini_test_temporal_v2.txt'
        elif train_test == "manual_test":
            file_name = 'mini_test_temporal_v2.txt'
            manual_file_name = 'manual_test/labeling_task/dic.dat'

        save_wiki_time_path = './origin_data/wiki_time_{}.txt'.format(train_test)

        vec_name = 'glove.txt'
        self.w_to_ix, self.w2v = process.read_in_vec(os.path.join(root, vec_name))
        # length of en2id > en_vecs
        self.en2id, self.en_vecs = process.read_in_en_vecs(os.path.join(root, "trained_vecs_50.npy"),
                                                  os.path.join(root, "entity2id.txt"))
        assert len(self.w_to_ix) == self.w2v.shape[0]
        # dict : [(en1_poss, en2_pos, [word,]),]


        # construct = True
        if construct:
            self.labels = process.create_labels()
            self.dict, self.rel_to_ix, self.natural, self.en2labels = process.construct_dataset(os.path.join(root,
                                                                                                         file_name),
                                                                                            self.labels,
                                                                                            self.w_to_ix,
                                                                                            train_test=train_test,
                                                                                            en2id=self.en2id,
                                                                                            save_wiki_time_path=save_wiki_time_path)

        else:
            self.dict, self.rel_to_ix, self.natural, self.en2labels = process.load_wiki_time(save_wiki_time_path,
                                                                                             self.w_to_ix,
                                                                                             en2id=self.en2id)

        # the length of en2id could be update in func : construct_dataset
        self.n_entity = len(self.en2id)
        self.key_list = list(self.dict.keys())

        self.vocab_size = self.w2v.shape[0]
        self.n_rel = len(self.rel_to_ix)
        self.max_sent_size = 50
        self.limit = 30
        # if position_embed:

        print("Vocab_size is:")
        print(self.vocab_size)
        print("Relation number is:")
        print(self.n_rel)
        print("# of bags:")
        print(self.__len__())

        if train_test == 'manual_test':
            self.replace_with_manual_label(os.path.join(root, manual_file_name))


    def __getitem__(self, index):
        # multi-instance learning
        # using bag as inputs
        ret = dict()
        en_pair = list(self.key_list)[index]

        # there are Mention objects in the bag
        bag = self.dict[en_pair]

        # ret['bag'] = bag
        bag, labels, ranks = self.extract_mentions(bag)
        tensor_bag = []
        for item in bag:
            tensor_bag.append(torch.cuda.LongTensor(item))

        tensor_labels = torch.cuda.LongTensor(labels)
        ret['bag'] = tensor_bag
        ret['label'] = tensor_labels
        ret['en_pair'] = en_pair
        ret['ranks'] = ranks

        return ret

    def __len__(self):
        return len(self.key_list)

    def collate_fn(self, data):
        return data

    def collate_bag_fn(self, data):
        for item in data:
            item['label'] = item['label'][-1].reshape(1)
                # .reshape(1, -1)
        return data

    def extract_mentions(self, mentions):
        ret = []
        labels = []
        ranks = []
        for item in mentions:
            sent_len = len(item.sent)
            pos1 = item.pos[0]
            pos2 = item.pos[1]
            con = []
            for j in range(sent_len):
                conl = pos1 - j + self.limit
                conl = max(0, conl)
                conl = min(self.limit * 2, conl)
                conr = pos2 - j + self.limit
                conr = max(0, conr)
                conr = min(self.limit * 2, conr)
                con.append([item.sent[j], conl, conr])
            ret.append(con)
            labels.append(item.tag)
            ranks.append(item.rank)

        return ret, labels, ranks

    def generate_manual_test_case(self, output_file, generate_length=200):
        # this function have to run construct dataset first. Or en2labels will be None.
        ids = list(range(len(self.key_list)))
        random.shuffle(ids)
        ids = ids[:generate_length]
        with open(output_file, 'w') as f:
            max_len = min(generate_length, len(self.key_list))
            for i in range(max_len):
                id = ids[i]
                key = self.key_list[id]
                mentions = self.dict[key]
                labels = self.en2labels[key]
                key_str = '\t'.join(mentions[0].en_pair_str)
                label_str = "\t".join(["{} : {}".format(label.time_str, label.relation) for label in labels])
                f.write(key_str + '\n')
                f.write("Ground Truth: \t" + label_str + '\n')
                for mention in mentions:
                    mention_str = '\t'.join([mention.time.time_str, ' '.join(mention.org_sent), mention.tag_name])
                    f.write(mention_str + '\n')
                f.write('###########\n')
        return

    def compute_manual_test_metric(self, file_path):
        with open(file_path, 'rb') as f:
            dic = pickle.load(f)
        count = 0
        tot = 0
        tagged = []
        manual = []
        escaped_counts = 0
        for en_pair, val in dic.items():
            en1, en2 = list(map(lambda x: self.en2id[x], en_pair))
            tagged_labels = [item.tag for item in self.dict[(en1, en2)]]
            # tagged_labels = [self.rel_to_ix[item] for item in tagged_labels]
            # relation ids
            mentions1 = [item.org_sent for item in self.dict[(en1, en2)]]
            mentions2 = [item[1] for item in val]
            manual_labels = [item[2] for item in val]
            i, j = 0, 0
            # for i in range(len(manual_labels)):
            while i < len(mentions1) and j < len(mentions2):
                # pdb.set_trace()
                if " ".join(mentions1[i]).strip() == mentions2[i].strip():
                    tot += 1
                    if manual_labels[j] == tagged_labels[i]:
                        count += 1
                    i += 1
                    j += 1
                else:
                    j += 1
                    escaped_counts += 1

            '''
            try:
                assert len(manual_labels) == len(tagged_labels)
            except:
                escaped_counts += len(manual_labels)
                print(en_pair)
                continue
            tagged += tagged_labels
            manual += manual_labels
            '''
        print("Dataset acc : {}".format(float(count) / tot))
        print("Escaped counts : {}".format(escaped_counts))

    def replace_with_manual_label(self, file_path):
        with open(file_path, 'rb') as f:
            dic = pickle.load(f)

        new_dict = defaultdict(list)
        escaped_counts = 0
        for en_pair, val in dic.items():
            en1, en2 = list(map(lambda x: self.en2id[x], en_pair))
            tagged_sents = self.dict[(en1, en2)]
            manual_labels = [item[2] for item in val]
            if len(tagged_sents) != len(manual_labels):
                escaped_counts += len(manual_labels)
                continue
            for i, item in enumerate(tagged_sents):
                item.tag = manual_labels[i]
            new_dict[(en1, en2)] = tagged_sents
        self.dict = new_dict
        self.key_list = list(self.dict.keys())

        return

if __name__ == '__main__':
    # wiki = WIKI_TIME('./data', train_test='test', construct=True)
    # out_dir = './manual_test'
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # wiki.generate_manual_test_case(os.path.join(out_dir, 'manual.test'))
    # wiki.compute_manual_test_metric('./manual_test/labeling_task/dic.dat')
    # wiki.generate_manual_test_case(os.path.join(out_dir, 'manual.test_all'), generate_length=10000)
    pass








