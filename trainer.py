from Dataset import Riedel_10, WIKI_TIME
from cnn_one import CNN_ONE
from cnn_att import CNN_ATT
from mem_cnn import MEM_CNN_RIEDEL, MEM_CNN_WIKI
from pcnn_one import PCNN_ONE
from pcnn_att import PCNN_ATT
from mem_pcnn import MEM_PCNN_RIEDEL
from cnn_ave import CNN_AVE
from tm_att import TM_ATT
from word_rel_mem import Word_Rel_MEM
from miml_conv import MIML_CONV, MIML_CONV_ATT, MIML_CONV_WORD_MEM_ATT

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import argparse
import os
import pdb
import numpy as np
import time
from tqdm import tqdm
from utils import precision_recall_compute_multi, one_hot, multi_hot_label


class Trainer():
    def __init__(self, config):
        root = config.dataset_dir
        self.problem = config.problem
        if self.problem == 'NYT-10':
            print('Reading Training data!')
            self.train_data = Riedel_10(root,
                                        debug=config.debug,
                                        use_whole_bag=config.use_whole_bag)
            print('Reading Testing data!')
            self.test_data = Riedel_10(root, train_test='test',
                                       debug=config.debug,
                                       use_whole_bag=config.use_whole_bag)
        else:
            print('Reading Training data!')
            self.train_data = WIKI_TIME(root,
                                        debug=config.debug)
                                        # use_whole_bag=config.use_whole_bag)
            print('Reading Testing data!')
            self.test_data = WIKI_TIME(root, train_test='test',
                                       debug=config.debug)
                                       # use_whole_bag=config.use_whole_bag)

        self.noise_and_clip = config.use_noise_and_clip

        self.batch_size = config.batch_size

        if config.problem == 'WIKI-TIME' and config.model != 'MEM_CNN_WIKI':
            collate_fn = self.train_data.collate_bag_fn
        else:
            collate_fn = self.train_data.collate_fn

        self.train_loader = data.DataLoader(self.train_data,
                                            batch_size=config.batch_size,
                                            pin_memory=False,
                                            shuffle=True,
                                            collate_fn=collate_fn)
        # print('Finish reading in train data!')
        self.test_loader = data.DataLoader(self.test_data,
                                            batch_size=config.batch_size,
                                            pin_memory=False,
                                            shuffle=False,
                                            collate_fn=collate_fn)
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
            'order_embed': config.order_embed,
            'circular': config.circular,
            # 'query_number': "once" if config.query_once else "all",
            'entity_pretrained_vecs':self.train_data.en_vecs,
            # 'max_hops':4,
            'max_hops':config.max_hops,
            'word_mem_hops' : config.word_mem_hops,
            'rel_mem_hops': config.rel_mem_hops,
            'remove_origin_query' : config.remove_origin_query,
            'memory_decay_weight' : config.memory_decay_weight,
            'query_type' : config.query_type,
            'order_weight': config.order_weight,
            'tri_attention' : config.tri_attention,
            'conv_type' : config.conv_type,
        }

        self.config = config
        # self.model = CNN_ONE(settings)
        # self.model_str = 'CNN_ONE'
        # self.model = CNN_ATT(settings)
        # self.model_str = 'CNN_ATT'
        models = {'CNN_ONE':CNN_ONE,
                  'CNN_ATT':CNN_ATT,
                  'MEM_CNN':MEM_CNN_RIEDEL,
                  'MEM_PCNN': MEM_PCNN_RIEDEL,
                  'MEM_CNN_WIKI': MEM_CNN_WIKI,
                  'PCNN_ONE':PCNN_ONE,
                  'PCNN_ATT':PCNN_ATT,
                  'CNN_AVE':CNN_AVE,
                  'TM_ATT':TM_ATT,
                  'WORD_REL_MEM':Word_Rel_MEM,
                  'MIML_CONV': MIML_CONV,
                  'MIML_CONV_ATT': MIML_CONV_ATT,
                  'MIML_CONV_WORD_MEM_ATT': MIML_CONV_WORD_MEM_ATT,
                  }
        model = models[config.model]
        self.model = model(settings)
        # self.model_str = 'MEM_CNN_RIEDEL'
        self.model_str = str(self.model.__class__.__name__)

        # load some of the pretrained weights
        # pre_model_path = '/data/yanjianhao/nlp/torch/torch_NRE/model/CNN_ATT_epoch_14'
        pre_model_path = './model/CNN_ATT_epoch_14'
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(pre_model_path)
        # filter different keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        # load the state dict
        # self.model.load_state_dict(model_dict)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.loss_func = nn.NLLLoss(size_average=False)
        # get a easier way to do this.
        self.binary_loss_models = {'Word_Rel_MEM', 'MIML_CONV', 'MIML_CONV_ATT', 'MIML_CONV_WORD_MEM_ATT'}
        if self.model_str in self.binary_loss_models:
            self.loss_func = self.binary_loss

        self.model_parameters = [item for item in self.model.parameters() if item.requires_grad]
        optimizer = {
            'sgd' : optim.SGD(iter(self.model_parameters), lr=config.lr),
            'adam' : optim.Adam(iter(self.model_parameters), lr=config.lr),
            'adadelta': optim.Adadelta(iter(self.model_parameters), lr=config.lr),
        }
        self.opt = optimizer[self.config.optimizer]

        self.start_epoch = 0
        self.max_epochs = config.max_epochs

        self.global_log = './experiment.log'
        self.timestamp = str(int(time.time()))
        if not config.debug:
            print('Logging!')
            self.log_experiment()
        for key, val in vars(self.config).items():
            print('\t\t{} : {}\n'.format(key, str(val)))

    def compute_loss(self, out, labels, epoch=0):
        # labels is list
        if self.problem == 'WIKI-TIME' and self.model_str == 'MEM_CNN_WIKI':
            labels = torch.cat(labels, dim=0)
            out = torch.cat(out, dim=0)
            loss = self.loss_func(out, labels)
        # out is a tuple
        elif self.model_str == 'TM_ATT':
            labels = torch.cat(labels)
            decay_period = 5
            decay = 0.9 ** (epoch // decay_period)
            alpha = decay
            beta = 0.1 * decay
            loss = alpha * self.loss_func(out[0], labels) + (1-alpha) * self.loss_func(out[1], labels) + beta * out[2]
        elif self.model_str in self.binary_loss_models:
            loss = self.loss_func(out, labels)
        # out is a tensor, label is a tensor
        else:
            labels = torch.cat(labels)
            loss = self.loss_func(out, labels)

        return loss

    def train(self):
        model_root = "./model/"
        for i in range(self.max_epochs):
            print('TimeStamp: {}'.format(self.timestamp))
            print('Epoch {}:'.format(i))
            acc_loss = 0
            batch_ix = 0
            pbar = tqdm(self.train_loader)
            for item in pbar:
                # somehow... input is wrapped with a list
                out = self.model(item)
                self.opt.zero_grad()
                # in training, there's only one label for each bag
                labels = [sample['label'] for sample in item]
                if self.model_str == 'TM_ATT':
                    loss = self.compute_loss(out, labels, epoch=i)
                    out = out[0]
                else:
                    loss = self.compute_loss(out, labels)
                loss.backward()
                if self.noise_and_clip:
                    # eta = 0.01
                    # disable noise gradien t
                    eta = 0.0
                    tau = 0.55
                    noise_stddev = eta / (1 + i) * tau
                    self._gradient_noise_and_clip(self.model.parameters(),
                                                  noise_stddev=noise_stddev, max_clip=config.max_clip)
                self._decay_learning_rate(self.opt, epoch=i)
                self.opt.step()
                # acc_loss += loss.data.item()

                # if (batch_ix + 1) % 50 == 0:
                #     print('Batch {}:'.format(batch_ix))
                #     print('Loss {}'.format(str(acc_loss / 50)))
                #     acc_loss = 0
                pbar.set_description("Loss: {}".format(loss))
                batch_ix += 1

            if self.problem == 'WIKI-TIME' and self.model_str == 'MEM_CNN_WIKI':
                self.evaluate_bag(epoch=i)
                self.evaluate_all(epoch=i)
            elif self.model_str in self.binary_loss_models:
                self.evaluate_binary(epoch=i)
            else:
                self.evaluate(epoch=i)
        model_path = os.path.join(model_root, '{}_{}_epoch_{}'.format(self.timestamp, self.model_str, str(self.max_epochs)))
        torch.save(self.model.state_dict(), model_path)


    def evaluate(self, epoch=None):
        saving_path = './results/' + self.model_str
        preds = []
        y_true = []
        self.model.eval()
        for batch_ix, item in enumerate(self.test_loader):
            labels = [each_item['label'] for each_item in item]
            out = self.model(item)

            labels = self._one_hot(labels, self.train_data.n_rel)
            try:
                assert labels.cpu().numpy().shape == out.cpu().detach().numpy().shape
            except:
                out = out.max(-1)[0]
            preds.append(out.cpu().detach().numpy())
            y_true.append(labels.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        precision, recall = precision_recall_compute_multi(y_true, preds)
        np.save(os.path.join(saving_path, '{}_Epoch_{}_precision.npy'.format(self.timestamp, epoch)), precision)
        np.save(os.path.join(saving_path, '{}_Epoch_{}_recall.npy'.format(self.timestamp, epoch)), recall)
        # self.train()
        self.model.train()

        return

    def evaluate_binary(self, epoch=None):
        '''
        In models with binary loss, they use sigmoid as the probability computing layer.
        So logits is enough to rank them.

        :param epoch: for which epoch, used for saving models.
        :return:
        '''
        assert self.model_str in self.binary_loss_models
        saving_path = './results/' + self.model_str
        self.model.eval()
        correct_num = 0
        correct_num_with_rel = 0
        tot = 0
        tot_with_rel = 0
        # pdb.set_trace()
        preds = []
        y_true = []
        for batch_ix, item in enumerate(self.test_loader):
            labels = [each_item['label'] for each_item in item]
            out = self.model(item)
            labels = multi_hot_label(labels, self.train_data.n_rel)
            preds.append(out.cpu().detach().numpy())
            y_true.append(labels.cpu().numpy())
            # remove NA
            labels = labels[:, 1:]
            out = out[:, 1:].view(-1, self.train_data.n_rel-1)
            # cast predictions
            pred = (out > 0.5).float()
            out_shape = out.shape
            if labels.shape != pred.shape:
                pdb.set_trace()
            correct_num += int(torch.sum((labels == pred)))
            correct_num_with_rel += int(torch.sum(labels * pred))
            tot += int(out_shape[0] * out_shape[1])
            tot_with_rel += int(torch.sum(labels))

        correct_num_without_rel = int(correct_num) - int(correct_num_with_rel)
        tot_without_rel = int(tot) - int(tot_with_rel)
        print('Total_with_rel:{}'.format(tot_with_rel))
        print('Total_without_rel:{}'.format(tot_without_rel))
        print('Correct_with_rel:{}'.format(correct_num_with_rel))
        print('Correct_without_rel:{}'.format(correct_num_without_rel))
        print("Precision_ALL : {}".format(float(correct_num)/tot))
        print("Precision_REL : {}".format(float(correct_num_with_rel) / tot_with_rel))
        preds = np.concatenate(preds, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        precision, recall = precision_recall_compute_multi(y_true, preds)
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
        np.save(os.path.join(saving_path, '{}_Epoch_{}_all_precision.npy'.format(self.timestamp, epoch)), precision)
        np.save(os.path.join(saving_path, '{}_Epoch_{}_all_recall.npy'.format(self.timestamp, epoch)), recall)
        # no saving now.
        self.model.train()


    def evaluate_bag(self, epoch=None):
        saving_path = './results/' + self.model_str
        preds = []
        y_true = []
        self.model.eval()
        for batch_ix, item in enumerate(self.test_loader):
            # labels : list of tensor shape []
            labels = [item['label'] for item in item]
            # out : list of tensor shape (bag_size, n_rel)
            out = self.model(item)
            out_bag = torch.stack([o[0] for o in out])
            labels = [label[-1] for label in labels]
            labels = self._one_hot(labels, self.train_data.n_rel)
            preds.append(out_bag.cpu().detach().numpy())
            y_true.append(labels.cpu().numpy())
            try:
                assert labels.cpu().numpy().shape == out_bag.cpu().detach().numpy().shape
            except:
                pdb.set_trace()
        preds = np.concatenate(preds, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        precision, recall = precision_recall_compute_multi(y_true, preds)
        np.save(os.path.join(saving_path, '{}_Epoch_{}_bag_precision.npy'.format(self.timestamp, epoch)), precision)
        np.save(os.path.join(saving_path, '{}_Epoch_{}_bag_recall.npy'.format(self.timestamp, epoch)), recall)
        # self.train()
        self.model.train()

        return

    def evaluate_all(self, epoch=None):
        saving_path = './results/' + self.model_str
        preds = []
        y_true = []
        self.model.eval()
        for batch_ix, item in enumerate(self.test_loader):
            labels = [item['label'] for item in item]
            out = self.model(item)
            # out_bag = [o[0] for o in out]
            # labels = [label[-1] for label in labels]
            labels = torch.cat(labels, dim=0)
            out = torch.cat(out, dim=0)
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
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
        np.save(os.path.join(saving_path, '{}_Epoch_{}_all_precision.npy'.format(self.timestamp, epoch)), precision)
        np.save(os.path.join(saving_path, '{}_Epoch_{}_all_recall.npy'.format(self.timestamp, epoch)), recall)
        # self.train()
        self.model.train()

        return

    # test for model
    def test(self):
        model_root = "./model/"
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

    def log_experiment(self):
        with open(self.global_log, 'a') as f:
            f.write("Experiment \n")
            f.write("\tTimestamp : {}\n".format(self.timestamp))
            for key, val in vars(self.config).items():
                f.write('\t\t{} : {}\n'.format(key, str(val)))
            f.write('\n')


    def _gradient_noise_and_clip(self, parameters,
                                 noise_stddev=1e-3, max_clip=40.0):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        nn.utils.clip_grad_norm(parameters, max_clip)
        # nn.utils.clip_grad_norm_(parameters, max_clip)

        for p in parameters:
            noise = torch.randn(p.size()) * noise_stddev
            if self.config.cuda:
                noise = noise.cuda()
            p.grad.data.add_(noise)

    def _decay_learning_rate(self, opt, epoch):
        decay_interval = self.config.decay_interval
        decay_ratio = self.config.decay_ratio

        decay_count = max(0, epoch // decay_interval)
        lr = self.config.lr * (decay_ratio ** decay_count)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        return lr

    def binary_loss(self, preds, targets, size_average=False):
        '''
            loss function defined in Feng-17's paper.
        '''
        targets = multi_hot_label(targets, self.train_data.n_rel)
        # remove NA
        '''
        preds = preds[:, 1:]
        targets = targets[:, 1:]
        # clamps within the min and max
        # follow the paper exactly
        preds = preds.view(-1, self.train_data.n_rel-1).clamp(1e-5, 1e5)
        preds = torch.abs(1 - targets - preds)
        log_preds = torch.log(preds)
        loss = torch.sum(log_preds)
        loss = -1 * loss
        '''
        # v2 --> BCE loss
        loss = F.binary_cross_entropy_with_logits(preds, targets, size_average=False)

        return loss

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = self.alpha * loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()






def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--riedel_10", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default='./origin_data/')
    parser.add_argument("--out_c", type=int, default=230)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--word_mem_hops", type=int, default=2)
    parser.add_argument("--rel_mem_hops", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.02/16)
    parser.add_argument("--decay_interval", type=int, default=10)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--max_clip", type=float, default=40.0)
    parser.add_argument("--word_embed_size", type=int, default=50)
    parser.add_argument("--max_sent_len", type=int, default=60)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    # parser.add_argument('--test_last_one', type=bool, default=False)
    parser.add_argument('--mem_version', type=int, default=1)
    parser.add_argument('--order_weight', type=float, default=1.0)
    parser.add_argument('--order_embed', action='store_true')
    parser.add_argument("--model", type=str, default='CNN_ATT')
    parser.add_argument('--circular', action='store_true')
    # parser.add_argument('--query_once', action='store_true')
    parser.add_argument("--memory_decay_weight", type=float, default=1)
    parser.add_argument("--query_type", type=str, default='RELATION')
    parser.add_argument('--remove_origin_query', action='store_true')
    parser.add_argument('--position_embedding', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--problem", type=str, default='NYT-10')
    parser.add_argument('--use_noise_and_clip', action='store_true')
    parser.add_argument('--use_whole_bag', action='store_true')
    parser.add_argument('--tri_attention', action='store_true')
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--conv_type", type=str, default='CNN')


    return parser.parse_args()


if __name__ == '__main__':
    config = parse_config()
    t = Trainer(config)
    t.train()
    # t.test()