import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Type, TypeVar

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from overrides import overrides

from nos.modules import LoadStateDictWithPrefix

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="FromParams")


class BaseModel(LoadStateDictWithPrefix, Model):
    def __init__(self,
                 vocab: Vocabulary):
        super().__init__(vocab)
        self.history: Dict[str, float] = defaultdict(float)
        self.batch_history: Dict[str, float] = defaultdict(float)
        self.sample_history: Dict[str, float] = defaultdict(float)
        self.json_metrics: Dict[str, list] = defaultdict(list)
        self.step_history: Dict[str, float] = defaultdict(float)
        self.epoch = 0

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, Any]:
        all_metrics: Dict[str, Any] = {}
        if self.history['_n_batches'] > 0:
            for metric, total in self.history.items():
                all_metrics[metric] = total

            for metric, total in self.batch_history.items():
                all_metrics[metric] = total / self.history['_n_batches']

            for metric, total in self.sample_history.items():
                all_metrics[metric] = total / self.history['_n_samples']

            for metric, total in self.step_history.items():
                all_metrics[metric] = total / self.history['_n_steps']

        if reset:
            self.history = defaultdict(float)
            self.batch_history = defaultdict(float)
            self.sample_history = defaultdict(float)
            self.step_history = defaultdict(float)
            if not self.training:
                self.epoch += 1

        return all_metrics

    def get_json_metrics(self, reset: bool = False) -> Dict[str, Any]:
        all_metrics: Dict[str, Any] = {}
        for metric, histogram in self.json_metrics.items():
            all_metrics[metric] = histogram

        if reset:
            self.json_metrics = defaultdict(list)

        return all_metrics

    @classmethod
    def from_params(cls: Type['BaseModel'],
                    vocab: Vocabulary,
                    params: Params,
                    constructor_to_call: Callable[..., T] = None,
                    constructor_to_inspect: Callable[..., T] = None,
                    **extras) -> 'BaseModel':
        logger.info(f"instantiating class {cls} from params "
                    f"{getattr(params, 'params', params)} and vocab {vocab}")

        model_dict = cls.get_params(vocab, params)
        params_dict = {**model_dict, **params.as_dict()}
        model = cls(vocab=vocab, **params_dict)

        return model

    @classmethod
    def get_params(cls, vocab: Vocabulary, params: Params) -> Dict[str, Any]:
        params_dict: Dict[str, Any] = {}

        params_dict['initializer'] = InitializerApplicator.from_params(
            params.pop('initializer', None))

        return params_dict

    def extend_embedder_vocab(self, embedding_sources_mapping: Dict[str, str] = None) -> None:
        """ Turn off vocab extension for now."""
