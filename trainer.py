from Dataset import Riedel_10
from cnn_one import CNN_ONE
from cnn_att import CNN_ATT
from mem_cnn import MEM_CNN_RIEDEL
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import argparse
import os
import pdb
import numpy as np
from utils import precision_recall_compute_multi

class Trainer():
    def __init__(self, config):
        root = config.dataset_dir
        print('Reading Training data!')
        self.train_data = Riedel_10(root)
        print('Reading Testing data!')
        self.test_data = Riedel_10(root, train_test='test')
        self.batch_size = config.batch_size

        self.train_loader = data.DataLoader(self.train_data,
                                            batch_size=config.batch_size,
                                            pin_memory=False,
                                            shuffle=True,
                                            collate_fn=self.train_data.collate_fn)
        # print('Finish reading in train data!')
        self.test_loader = data.DataLoader(self.test_data,
                                            batch_size=config.batch_size,
                                            pin_memory=False,
                                            shuffle=False,
                                            collate_fn=self.train_data.collate_fn)
        # print('Finish reading in test data!')

        settings = {
            "use_cuda": config.cuda,
            "vocab_size": self.train_data.vocab_size,
            "word_embed_size": 50,
            "sentence_size": config.max_sent_len,
            "n_rel": self.train_data.n_rel,
            "word_embeds": self.train_data.w2v,
            "out_c": config.out_c,
            "dropout_p": config.dropout_p,
            'position_embedding': config.position_embedding,
            'pos_embed_size' : 5,
            'pos_limit': self.train_data.limit,
            # 'n_entity': self.train_data.n_entity,
            # 'entity_pretrained_vecs': self.train_data.en_vecs,
            'entity_embed_size': 50,
            # 'n_entity': len(self.train_data.en2id),
            # 'max_hops': config.max_hops,
            'version': config.mem_version,
            # 'delete_order_embed': config.delete_order_embed,
            # 'circular': config.circular,
            # 'query_number': "once" if config.query_once else "all",
            # 'entity_query': config.entity_query
            'entity_pretrained_vecs':self.train_data.en_vecs,
            'max_hops':4,
        }

        # self.model = CNN_ONE(settings)
        # self.model_str = 'CNN_ONE'
        # self.model = CNN_ATT(settings)
        # self.model_str = 'CNN_ATT'
        self.model = MEM_CNN_RIEDEL(settings)
        self.model_str = 'MEM_CNN_RIEDEL'
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.loss_func = nn.NLLLoss(size_average=False)
        self.opt = optim.SGD(self.model.parameters(), lr=0.02/16)

        self.start_epoch = 0
        self.max_epochs = 15

    def train(self):
        model_root = "/data/yanjianhao/nlp/torch/torch_NRE/model/"
        for i in range(self.max_epochs):
            print('Epoch {}:'.format(i))
            for batch_ix, item in enumerate(self.train_loader):
                # somehow... input is wrapped with a list
                out = self.model(item)
                self.opt.zero_grad()
                # in training, there's only one label for each bag
                labels = [sample['label'] for sample in item]
                labels = torch.cat(labels)

                loss = self.loss_func(out, labels)
                loss.backward()
                self.opt.step()

                if (batch_ix + 1) % 50 == 0:
                    print('Batch {}:'.format(batch_ix))
                    print('Loss {}'.format(str(loss)))
            model_path = os.path.join(model_root, '{}_epoch_{}'.format(self.model_str, str(i)))
            torch.save(self.model.state_dict(), model_path)
            self.evaluate(epoch=i)

    def evaluate(self, epoch=None):
        saving_path = '/data/yanjianhao/nlp/reproduce/results/' + self.model_str
        preds = []
        y_true = []
        self.model.eval()
        for batch_ix, item in enumerate(self.test_loader):
            bags, labels = item
            out = self.model(bags, labels)
            labels = self._one_hot(labels, self.train_data.n_rel)
            preds.append(out.cpu().detach().numpy())
            y_true.append(labels.cpu().numpy())
            try:
                assert labels.cpu().numpy().shape == out.cpu().detach().numpy().shape
            except:
                pdb.set_trace()
        preds = np.concatenate(preds, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        precision, recall = precision_recall_compute_multi(y_true, preds)
        np.save(os.path.join(saving_path, 'Epoch_{}_precision.npy'.format(epoch)), precision)
        np.save(os.path.join(saving_path, 'Epoch_{}_recall.npy'.format(epoch)), recall)
        # self.train()
        self.model.train()

        return

    # test for model
    def test(self):
        model_root = "/data/yanjianhao/nlp/torch/torch_NRE/model/"
        for i in range(self.max_epochs):
            print('For epoch {}'.format(i))
            model_path = os.path.join(model_root, '{}_epoch_{}'.format(self.model_str, str(i)))
            self.model.load_state_dict(torch.load(model_path))
            self.evaluate(epoch=i)

    # ids can be list-shape object
    # deal with multi-label scenario
    def _one_hot(self, ids, n_rels):
        bz = len(ids)
        labels = torch.zeros(bz, n_rels)
        for ix, id in enumerate(ids):
            labels[ix, id] = 1
        return labels




def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--riedel_10", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default='/data/yanjianhao/nlp/torch/torch_NRE/origin_data/')
    parser.add_argument("--out_c", type=int, default=230)
    # parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--decay_interval", type=int, default=25)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--max_clip", type=float, default=40.0)
    parser.add_argument("--word_embed_size", type=int, default=50)
    parser.add_argument("--max_sent_len", type=int, default=60)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    # parser.add_argument('--test_last_one', type=bool, default=False)
    parser.add_argument('--mem_version', type=int, default=1)
    # parser.add_argument('--delete_order_embed', action='store_true')
    # parser.add_argument('--circular', action='store_true')
    # parser.add_argument('--query_once', action='store_true')
    # parser.add_argument('--entity_query', action='store_true')
    parser.add_argument('--position_embedding', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_config()
    t = Trainer(config)
    t.train()
    # t.test()