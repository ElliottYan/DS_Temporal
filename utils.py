import numpy as np
import pdb
import torch
import gc

# create one-hot
# not needed when labels are already one-hot
def one_hot(ids, n_rel):
    """
    ids: numpy array or list shape:[batch_size,]
    n_rel: # of relation to classify
    """
    labels = torch.zeros((ids.shape[0], n_rel)).cuda()
    labels[torch.arange(ids.shape[0]).long(), ids.long()] = 1
    return labels


def multi_hot_label(ids, n_rel):
    """
    ids: list shape:[batch_size, m]
    n_rel: # of relation to classify
    """
    labels = torch.zeros((len(ids), n_rel)).cuda()
    for i in range(len(labels)):
        labels[torch.Tensor([i]*ids[i].shape[0]).long().cuda(), ids[i]] = 1
    return labels


def precision_recall_compute_multi(labels, y_pred):
    labels = labels[:, 1:].reshape(-1)
    y_pred = y_pred[:, 1:].reshape(-1)

    idx = np.argsort(y_pred)

    tot = np.sum(labels)
    y_pred = y_pred[idx][-2000:]
    labels = labels[idx][-2000:]

    correct = 0
    # tot = torch.sum(labels)
    precision_hat = []
    recall_hat = []
    for i in range(2000):
        correct += labels[2000 - i - 1]
        precision = correct / (i + 1)
        recall = correct / tot
        #     print(precision, recall)
        precision_hat.append(precision)
        recall_hat.append(recall)
        if i % 200 == 0:
            print('Precision: {}'.format(precision))
            print('Recall: {}'.format(recall))
    print('Correct: {}'.format(correct))
    precision_hat = np.array(precision_hat)
    recall_hat = np.array(recall_hat)
    return precision_hat, recall_hat

def compute_max_f1(precision, recall):
    max_f1 = 0
    for k in range(len(precision)):
        f1 = 2 * (precision[k] * recall[k]) / (precision[k] + recall[k])
        max_f1 = max([max_f1, f1])
    return max_f1


def compute_average_f1(precision, recall):
    sum = 0
    for k in range(len(precision)):
        f1 = 2 * (precision[k] * recall[k]) / (precision[k] + recall[k])
        sum += f1
    return float(sum) / len(precision)


def logging_existing_tensor():
    ret = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
                ret.append(obj)
        except:
            pass
    return ret
