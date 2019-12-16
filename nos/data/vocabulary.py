from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

from allennlp.common import Params
from allennlp.data import instance as adi  # pylint: disable=unused-import
from allennlp.data.vocabulary import DEFAULT_NON_PADDED_NAMESPACES, Vocabulary


@Vocabulary.register('empty')
class EmptyVocabulary(Vocabulary):
    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))

        return cls(counter=namespace_token_counts,
                   non_padded_namespaces=DEFAULT_NON_PADDED_NAMESPACES)

    @classmethod
    def from_instances(cls,
                       instances: Iterable['adi.Instance'],
                       min_count: Dict[str, int] = None,
                       max_vocab_size: Union[int, Dict[str, int]] = None,
                       non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
                       pretrained_files: Optional[Dict[str, str]] = None,
                       only_include_pretrained_words: bool = False,
                       tokens_to_add: Dict[str, List[str]] = None,
                       min_pretrained_embeddings: Dict[str, int] = None) -> 'Vocabulary':
        """
        Constructs a vocabulary given a collection of `Instances` and some parameters.
        We count all of the vocabulary items in the instances, then pass those counts
        and the other parameters, to :func:`__init__`.  See that method for a description
        of what the other parameters do.
        """
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))

        return cls(counter=namespace_token_counts,
                   min_count=min_count,
                   max_vocab_size=max_vocab_size,
                   non_padded_namespaces=non_padded_namespaces,
                   pretrained_files=pretrained_files,
                   only_include_pretrained_words=only_include_pretrained_words,
                   tokens_to_add=tokens_to_add,
                   min_pretrained_embeddings=min_pretrained_embeddings)

    def __repr__(self) -> str:
        return 'EmptyVocabulary()'

    def __str__(self) -> str:
        return 'EmptyVocabulary()'
