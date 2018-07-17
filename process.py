"""
This file is used to process Downloaded pretrained embeddings from http://openke.thunlp.org/index/toolkits#pretrained-embeddings
"""
import numpy as np
import os
import pdb

en2id_path = '/data/yanjianhao/tmp/Freebase/knowledge graphs/entity2id.txt'
en_vecs_path = '/data/yanjianhao/tmp/Freebase/embeddings/dimension_50/transe/entity2vec.bin'

data_path = '/data/yanjianhao/nlp/torch/torch_NRE/data'


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
    np.save('/data/yanjianhao/nlp/torch/torch_NRE/data/en_vecs/nyt_en_vecs.npy', new_vecs)
    with open('/data/yanjianhao/nlp/torch/torch_NRE/data/en_vecs/en2id.txt', 'w') as f:
        for key, val in entity_id.items():
            f.write('{}\t{}\t{}\n'.format(key, entity2str[key], str(val)))


if __name__ == '__main__':
    main()




