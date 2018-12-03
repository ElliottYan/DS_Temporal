import torch
import torch.nn.functional as F
import torch.nn as nn
from cnn import CNN
from word_rel_mem import Word_MEM

class CNN_WORD_MEM(CNN):
    def __init__(self, settings):
        super(CNN_WORD_MEM, self).__init__(settings)
        self.word_mem = Word_MEM(self.input_size, settings)

    def _create_sentence_embedding(self, bags, labels):
        batch_features = []
        for ix, bag in enumerate(bags):
            # pdb.set_trace()
            label = labels[ix]
            features = []
            for item in bag:
                w2v = self.w2v(item.t()[0])
                # this may need some modification for further use.
                pos1 = self.pos1_embed(item[:, 1])
                pos2 = self.pos2_embed(item[:, 2])
                word_feature = torch.cat([w2v, pos1, pos2], dim=-1).unsqueeze(0).unsqueeze(0)
                # feature = torch.cat([w2v, pos1, pos2], dim=-1).unsqueeze(0).unsqueeze(0)
                feature = self.conv(word_feature).squeeze(-1)
                feature = (F.max_pool1d(feature, feature.size(-1)).squeeze(-1) + self.conv_bias).reshape(1, -1)

                # mem_feature (with position embedding)
                word_mem_feat = self.word_mem({
                    'w2v': word_feature.reshape(-1, self.input_size),
                    'sent': item,
                }).reshape(1, -1)  # 1 * feature_size

                feature = torch.cat([feature, word_mem_feat], dim=-1)  # 1 * (feature_size + out_c)

                # this tanh is little different from lin-16's.
                feature = self.tanh(feature)
                # feature = self.dropout(feature)
                # dropout is a little different too.
                features.append(feature)
            features = torch.cat(features, dim=0)
            batch_features.append(features)
        return batch_features


