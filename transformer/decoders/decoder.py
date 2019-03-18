import torch
import torch.nn as nn

# from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
# from onmt.modules import context_gate_factory, GlobalAttention
# from onmt.utils.rnn_factory import rnn_factory

# from onmt.utils.misc import aeq


class DecoderBase(nn.Module):
    """Abstract class for decoders.

    Args:
        attentional (bool): The decoders returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor.

        Subclasses should override this method.
        """

        raise NotImplementedError

