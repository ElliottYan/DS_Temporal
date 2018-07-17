import numpy as np
import pdb

# create one-hot
# not needed when labels are already one-hot
def one_hot(ids, n_rel):
    """
    ids: numpy array or list shape:[batch_size,]
    n_rel: # of relation to classify
    """
    labels = np.zeros((ids.shape[0], n_rel))
    labels[np.arange(ids.shape[0]), ids] = 1
    return labels

# n_rel = 53
# labels = one_hot(y_true, n_rel)
# print(labels.shape)


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
        if i % 100 == 0:
            print('Precision: {}'.format(precision))
            print('Recall: {}'.format(recall))
    print('Correct: {}'.format(correct))
    precision_hat = np.array(precision_hat)
    recall_hat = np.array(recall_hat)
    return precision_hat, recall_hat

