"""
This file is used to process Downloaded pretrained embeddings from http://openke.thunlp.org/index/toolkits#pretrained-embeddings
"""
import numpy as np
import os
import pdb
from sklearn.preprocessing import normalize

import pandas as pd
import pickle
# from pymongo import MongoClient
from collections import defaultdict
from structures import time_signature, Stack, Mention
import random, math
import pdb
import re, string, time, os
from functools import wraps

from itertools import product

import spacy
from spacy.symbols import ORTH, LEMMA, POS
# nlp = spacy.load('en', disable=['ner'])
data_root = './data/'
# special_case = [{ORTH: u'TIME'},]
# nlp.tokenizer.add_special_case(u'TIME', special_case)

en2id_path = './Freebase/knowledge graphs/entity2id.txt'
en_vecs_path = './Freebase/embeddings/dimension_50/transe/entity2vec.bin'

data_path = './data'


# the overall en2id
def read_in_en2id():
    print('Read in en2id')
    count = 0
    with open(en2id_path, 'r') as f:
        num = int(f.readline())
        en2id = {}
        for i in range(num):
            key, val = f.readline().strip().split()
            en2id[key] = int(val)
            count += 1
            if count % 1000 == 0:
                print('Now process {} items'.format(count))
        return en2id


def filter_by_dataset():
    entity_set = set()
    entity_id = {}
    entity2str = dict()
    with open(os.path.join(data_path, 'train.txt'), 'r') as f:
        train = f.readlines()
    with open(os.path.join(data_path, 'test.txt'), 'r') as f:
        test = f.readlines()
    count = 0
    total = train + test
    for line in total:
        tmp = line.strip().split()
        e1 = tmp[0]
        e2 = tmp[1]
        e1_str = tmp[2]
        e2_str = tmp[3]
        if e1 not in entity_set:
            entity_id[e1] = len(entity_id)
            entity_set.add(e1)
            entity2str[e1] = e1_str
        if e2 not in entity_set:
            entity_id[e2] = len(entity_id)
            entity_set.add(e2)
            entity2str[e2] = e2_str

        count += 1
        if count % 1000 == 0:
            print('Now process {} lines'.format(count))

    assert len(entity2str) == len(entity_set)

    return entity_set, entity_id, entity2str




# in py3 version
def fn_timer(func):
    @wraps(func)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (func.__name__, str(t1 - t0))
               )
        return result

    return function_timer


def set_default():
    return None


def clean(row):
    if type(row) != str:
        #         print(row)
        return 'NaN'
    elif len(row) < 20:
        return 'NaN'
    else:
        return row.split('T')[0]


def check_relation(label, x):
    # todo : need to add negation to each label.
    # label : a list of time_signature
    # x : a query for its label
    stack = Stack()
    unpop = []
    modified = True
    for node in label:
        if node < x:
            if node.type == 'start':
                stack.push(node)
                # print('push')
            if node.type == 'end':
                if node.relation == stack.peek().relation:
                    stack.pop()
                    # print('pop')
                    while(unpop and unpop[-1] == stack.peek().relation):
                        stack.pop()
                        # print('pop')
                        unpop = unpop[:-1]
                else:
                    unpop.append(node.relation)
        else:
            if not modified:
                if node.type == 'end' and stack.peek().relation[:3] == 'NOT' and node.relation[:3] != 'NOT':
                    stack.push(time_signature('0000-00-00', relation=node.relation))
                if stack.size() == 1:
                    rel = node.relation
                else:
                    rel = stack.peek().relation
                return rel
            else:
                rel = stack.peek().relation
                return rel


def create_labels():
    with open(data_root + "alignment.dat", 'rb') as f:
        # align is the map from wiki-data to wiki-pedia
        align = pickle.load(f)
    with open(data_root + "r_synonym.dat", 'rb') as f:
        r_synonym = pickle.load(f)

    entities_pair = pd.read_csv(data_root + "entities.csv")

    formal_entities_pair = pd.concat([entities_pair[['entity1','entity2', 'entity1Label', 'entity2Label', 'relation_name']],
                                      entities_pair['start_time'].apply(clean) ,entities_pair['end_time'].apply(clean)], axis=1)

    # This is for creating label sequences
    labels = defaultdict(list)
    # pdb.set_trace()
    for ix, row in formal_entities_pair.iterrows():
        # # Maybe there is no alignment for entity in wiki-data
        # try:
        #     en1 = align[row['entity1']]
        # except KeyError:
        #     en1 = row['entity1Label']
        # try:
        #     en2 = align[row['entity2']]
        # except KeyError:
        #     en2 = row['entity2Label']

        # should not use any alignment in this process.
        en1 = row['entity1Label']
        en2 = row['entity2Label']

        # bug is in r_synonym
        # en1_label =

        en1 = Normalization(en1)
        en2 = Normalization(en2)
        rel = "_".join(row['relation_name'].split())

        if (en2, en1) in labels.keys():
            # exchange en1 & en2
            en1, en2 = en2, en1

    #   initialization for labels
    #   each time signature denotes the end of some relation
    #     if not labels[(en1, en2)]:
    #         labels[(en1, en2)].append(time_signature('0000-00-00', relation='NA', node_type='start'))
    #         labels[(en1, en2)].append(time_signature('9999-99-99', relation='NA', node_type='end'))
        if row['start_time'] != 'NaN':
            labels[(en1, en2)].append(time_signature(row['start_time'], relation=rel, node_type='start'))
        if row['end_time'] != 'NaN':
            labels[(en1, en2)].append(time_signature(row['end_time'], relation=rel, node_type='end'))
    for key, item in labels.items():
        item.sort()
        # add at last
        start_rel = item[0].relation
        end_rel = item[-1].relation
        # todo: check its effect
        item.insert(0, time_signature('0000-00-00', relation='NOT_' + start_rel, node_type='start'))
        # item.insert(0, time_signature('0000-00-00', relation='NA', node_type='start'))
        item.append(time_signature('9999-99-99', relation='NOT_' + end_rel, node_type='end'))

    with open('./data/labels.dat', 'wb') as f:
        pickle.dump(labels, f)
    print('Label making done!')
    return labels
    # print(label for label in labels[:10])


def unit_test(labels):
    # This is unit test
    label = labels[('Euro', 'Estonia')]
    label.append(time_signature('2012-01-01', relation='test'))
    label.append(time_signature('2015-01-01', relation='currency', node_type='end'))
    label.sort()

    print([(t.time, t.relation, t.type) for t in labels[('Euro', 'Estonia')]])

    tmp1 = time_signature('_'.join(['2015', '01', '01']), relation='NA', node_type='mention')
    tmp2 = time_signature('_'.join(['2015', '11', '01']), relation='NA', node_type='mention')
    label = labels[('Euro', 'Estonia')]

    print(check_relation(label, tmp1))
    print(check_relation(label, tmp2))


def construct_dataset(file_path, labels, w_to_ix, train_test='train', en2id=None):
    # import reverse synonym
    # here we got doing the mapping in dataset construction phase
    with open('./origin_data/r_synonym.dat', 'rb') as f:
        r_synonym = pickle.load(f)
    print('Reading reverse synonym done!')

    # read-in rel_to_ix(modified version)
    rel_57 = True
    if rel_57:
        rel2ix_path = "./origin_data/rel2ix_temporal.txt"
    else:
        rel2ix_path = "./origin_data/rel2ix_temporal_v2.txt"
    rel_to_ix = defaultdict(set_default)
    with open(rel2ix_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        tmp = line.split()
        rel, ix = "_".join(tmp[:-1]), int(tmp[-1])
        rel_to_ix[rel] = ix
    # rel_to_ix['PAD'] = len(rel_to_ix)
    print('Reading rel_to_ix done!')

    mentions = defaultdict(list)
    natural = defaultdict(list)
    en2labels = defaultdict(list)
    mention_filter = defaultdict(set)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    debug = False
    outputs = dict()
    count = 0
    for line in lines:
        # count += 1
        # if count > 5:
        #     break
        line = line.split(',', maxsplit=8)
        # print(line)
        # extract all infos from train.txt

        # rel, en1, en2, pos1, pos2 = line[1:6]
        # year, month, day = line[6:9]
        # sent = line[9].split()
        en1, en2, pos1, pos2 = line[1:5]
        year, month, day = line[5:8]
        sent = line[8].split()

        # from mentions synonym -> entity label
        # considering multi-mapping
        # en1_list = r_synonym[en1]
        # en2_list = r_synonym[en2]
        en1_list = [en1,]
        en2_list = [en2,]

        # outputs.append(str([en1, en2]) + " : \n")

        for en1, en2 in product(en1_list, en2_list):
            if tuple(sent) in mention_filter[(en1, en2)]:
                continue

            #   swap in case en1 and en2 's order may differ
            if labels[(en2, en1)]:
                en1, en2 = en2, en1
            if not labels[(en1, en2)]:
                # pdb.set_trace()
                continue
                # pass
            en2label = labels[(en1, en2)]
            outputs[(en1, en2)] = []
            tmp = time_signature("-".join([year, month, day]), node_type='mention')
            # pdb.set_trace()
            if (en1, en2) == ('netherlands', 'dries_van_agt'):
                # pdb.set_trace()
                continue
            tag = check_relation(en2label, tmp)

            if tag not in rel_to_ix.keys():
                # rel_to_ix[tag] = len(rel_to_ix) - 1
                tag = 'NA'
            # turn tag into int
            tag_name = tag
            tag = rel_to_ix[tag]
            # adding for understand test cases
            # here : sent is words, tagname is relation label
            natural[(en1, en2)].append(Mention(tag_name, tmp, sent))

            # mentions[(en1, en2)].append()
            org_sent = sent
            if not debug:
                sent = [w_to_ix[word] if w_to_ix[word] else w_to_ix['UNK'] for word in sent]
            # mentions.append((pos1, pos2, sent, year, month, day, tag))
            en_pair_str = (en1, en2)
            if en2id:
                if en1 not in en2id.keys():
                    en2id[en1] = len(en2id)
                elif en2 not in en2id.keys():
                    en2id[en2] = len(en2id)
                # else:
                en1, en2 = en2id[en1], en2id[en2]
            count += 1
            mention_filter[(en1, en2)].add(tuple(sent))
            en2labels[(en1, en2)] = en2label
            mentions[(en1, en2)].append(Mention(sent, en_pair_str=en_pair_str, org_sent=org_sent, tag=tag, tag_name=tag_name, pos1=int(pos1), pos2=int(pos2), time=tmp))

    print('mention count : {}'.format(count))
    # keep mentions sorted
    for key, item in mentions.items():
        item.sort()
        rank = 0
        for i in range(1, len(item)):
            if item[i].time == item[i-1].time:
                item[i].rank = rank
            else:
                rank += 1
                item[i].rank = rank
    print('Finish create labels!')

    if debug:
        output_lines = []
        used = set()
        count = 0
        for en_pair in outputs.keys():
            prev = None
            if (en_pair[1], en_pair[0]) in used:
                continue
            used.add(en_pair)
            en1,en2 = en2id[en_pair[0]], en2id[en_pair[1]]
            tmp = mentions[(en1, en2)] + mentions[(en2, en1)]
            output_lines.append(str(en_pair) + ":\n")
            for mention in tmp:
                output_lines.append(mention.tag_name + '\t' + str(mention.time.time) + " : \n")
                if prev and prev != mention.tag_name:
                    count += 1
                    # print(prev, mention.tag_name)
                    if prev[4:] != mention.tag_name and mention.tag_name[4:] != prev:
                        print(prev, mention.tag_name)
                prev = mention.tag_name
                try:
                    output_lines.append(" ".join(mention.sent) + "\n")
                except:
                    pdb.set_trace()
                    pass
            output_lines.append('\n')
        print(count)

        with open('./origin_data/en+label+sent.txt', 'w') as f:
            f.writelines(output_lines)
        print('Writing to outputs!')

    with open("./origin_data/mentions_" + train_test + ".dat", 'wb') as fout:
        pickle.dump(mentions, fout)

    print('Finish save intermediate results! ')
    # pdb.set_trace()
    return mentions, rel_to_ix, natural, en2labels


def Normalization(s):
    # s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"\S", r" ", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = s.lower().strip()
    s = re.sub(r"(\s)", r"_", s)
    return s


def tokenization(sent):
    # CAPITAL WORD TIME does not appear in text
    nlp = spacy.load('en', disable=['ner'])
    special_case = [{ORTH: u'TIME'},]
    nlp.tokenizer.add_special_case(u'TIME', special_case)
    sent = re.sub(r'<t>', r'TIME', sent.lower())
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sent = regex.sub(" ", sent)
    sent = nlp(sent)
    w = []
    for word in sent:
        w.append(word.text)
    return " ".join(w)


def separate_datasets():
    # need +ion from train to test

    # with open(data_root + '/origin_data/mentions2/2018_01_24/mentions.csv', 'r') as f:
    #     lines = f.readlines()[1:]
    with open(data_root + '/origin_data/mentions4/2018_04_27/mentions.csv', 'r') as f:
        lines = f.readlines()

    with open('./origin_data/r_synonym.dat', 'rb') as f:
        r_synonym = pickle.load(f)
    mentions_count = defaultdict(int)
    mentions = defaultdict(list)
    print('extracting mentions counts from csv file')
    for l in lines:
        # count += 1
        # if count > 20:
        #     break

        line = l.split(',', maxsplit=8)

        # test for de-bug
        if l[0] == '30957':
            pdb.set_trace()

        # extract all infos from train.txt
        en1, en2, pos1, pos2 = line[1:5]
        year, month, day = line[5:8]
        sent = line[8]
        # extract all infos from train.txt
        # if labels[(en2, en1)]:
        #     en1, en2 = en2, en1
        key_list = mentions.keys()
        if (en2, en1) in key_list:
            en1, en2 = en2, en1

        mentions_count[(en1, en2)] += 1

        line[8] = sent
        mentions[(en1, en2)].append(",".join(line))

    tmp = list(mentions.keys())
    # random.shuffle is in-place operation
    random.shuffle(tmp)
    train_rate = 0.7
    train = tmp[:math.floor(train_rate * len(tmp))]
    test = tmp[math.floor(train_rate * len(tmp)):]
    sum = 0
    for i in train:
        sum += mentions_count[i]
    print('There are %d mentions sentence in train!'%sum)
    train_sents = []
    test_sents = []
    for i in train:
        train_sents += mentions[i]
    for i in test:
        test_sents += mentions[i]
    with open("./data/train_temporal_v2.txt", 'w') as fout:
        fout.writelines(train_sents)
    with open("./data/test_temporal_v2.txt", 'w') as fout:
        fout.writelines(test_sents)


def test_dataset(file_path):
    # import reverse synonym
    # here we got doing the mapping in dataset construction phase
    with open('origin_data/r_synonym.dat', 'rb') as f:
        r_synonym = pickle.load(f)
    print('Reading reverse synonym done!')

    # read-in rel_to_ix(modified version)
    rel_to_ix = defaultdict(set_default)
    with open("./origin_data/rel2ix_temporal.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        tmp = line.split()
        rel, ix = " ".join(tmp[:-1]), int(tmp[-1])
        rel_to_ix[rel] = ix
    # rel_to_ix['PAD'] = len(rel_to_ix)
    print('Reading rel_to_ix done!')

    mentions = defaultdict(list)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # count += 1
        # if count > 5:
        #     break
        line = line.split(',', maxsplit=9)
        # print(line)
        # extract all infos from train.txt
        rel, en1, en2, pos1, pos2 = line[1:6]
        year, month, day = line[6:9]
        sent = line[9].split()

        # from mentions synonym -> entity label
        # considering multi-mapping
        en1_list = r_synonym[en1]
        en2_list = r_synonym[en2]

        for en1, en2 in product(en1_list, en2_list):

            #   swap in case en1 and en2 's order may differ
            if mentions[(en2, en1)]:
                en1, en2 = en2, en1
            mentions[(en1, en2)].append(sent)

    count = 0
    for key, item_list in mentions.items():
        if len(item_list) > 50:
            print(item_list)
            count += 1

    print(count)

    return

def create_mini_dataset(in_file_path, out_file_path):
    entities_pair = pd.read_csv('./origin_data/entities.csv')
    pairs = set()
    for ix, row in entities_pair.iterrows():
        en1 = Normalization(row['entity1Label'])
        en2 = Normalization(row['entity2Label'])
        pairs.add((en1, en2))
        # pairs.add((en2, en1))
    with open(in_file_path, 'r') as f:
        lines = f.readlines()

    outputs = []

    for line in lines:
        # pdb.set_trace()
        tmp = line.strip().split(',', maxsplit=8)
        en1 = tmp[1]
        en2 = tmp[2]
        if (en1, en2) in pairs or (en2, en1) in pairs:
            outputs.append(line)

    with open(out_file_path, 'w') as fout:
        fout.writelines(outputs)


def main():
    # 50d vectors
    vecs = np.memmap(en_vecs_path, dtype='float32', mode='r').reshape(-1, 50)
    en2id = read_in_en2id()
    entity_set, entity_id, entity2str = filter_by_dataset()
    new_vecs = np.zeros((len(entity_id), 50))
    count = 0
    for key, val in entity_id.items():
        # entity = entity2str[key]
        try:
            vec = vecs[en2id[key]]
        except:
            # sometimes it does not contain
            count += 1
            vec = np.random.rand(50)
        new_vecs[val] = vec
    print('No pretrained embeddings: {} items'.format(count))

    # saving module
    np.save('./data/en_vecs/nyt_en_vecs.npy', new_vecs)
    with open('./data/en_vecs/en2id.txt', 'w') as f:
        for key, val in entity_id.items():
            f.write('{}\t{}\t{}\n'.format(key, entity2str[key], str(val)))

# w2v --->   word : ndarray
def read_in_vec(path):
    D = 50
    w_to_ix = defaultdict(set_default)
    with open(path, 'r')as f:
        lines = f.readlines()
    vecs = []
    for ix, line in enumerate(lines[1:]):
        splits = line.split()
        # convert text to numpy array
        vec = np.array([float(n) for n in splits[1:]])
        if w_to_ix[splits[0]]:
            pdb.set_trace()
        w_to_ix[splits[0]] = ix
        vecs.append(vec)
    w_to_ix['TIME'] = len(vecs)
    # special token for time expressions.
    w_to_ix['UNK'] = len(vecs) + 1
    # adding for special tokens [UNK & TIME]
    vecs.append(np.random.randn(D))
    vecs.append(np.random.randn(D))

    # vecs -> vocab_size * D
    vecs = np.stack(vecs, axis=1).T
    vecs = normalize(vecs)
    return w_to_ix, vecs


def read_in_en_vecs(en_vec_path, en2id_path):
    vecs = np.load(en_vec_path)
    with open(en2id_path, 'r') as f:
        lines = f.readlines()
    en2id = {}
    for line in lines:
        key,val = line.strip().split()
        en2id[key] = int(val)
    return en2id, vecs

def set_default():
    return None

if __name__ == '__main__':
    main()




