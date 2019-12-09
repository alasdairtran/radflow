import logging

import torch.nn as nn
from allennlp.common.registrable import Registrable

from nos.modules.mixins import LoadStateDictWithPrefix

logger = logging.getLogger(__name__)


class Decoder(LoadStateDictWithPrefix, Registrable, nn.Module):
    def get_output_dim(self) -> int:
        raise NotImplementedError()


class DecoderLayer(Registrable, nn.Module):
    pass
