

import os
from os import PathLike
from typing import List, Union

import yaml
from allennlp.commands.train import train_model
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params, parse_overrides, with_fallback
from allennlp.models import Model


def train_model_from_file(parameter_filename: Union[str, PathLike],
                          serialization_dir: Union[str, PathLike],
                          overrides: str = "",
                          recover: bool = False,
                          force: bool = False,
                          node_rank: int = 0,
                          include_package: List[str] = None,
                          dry_run: bool = False,
                          file_friendly_logging: bool = False) -> Model:
    """
    A wrapper around [`train_model`](#train_model) which loads the params from a file.
    # Parameters
    parameter_filename : `str`
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : `str`
        The directory in which to save results and logs. We just pass this along to
        [`train_model`](#train_model).
    overrides : `str`
        A JSON string that we will use to override values in the input parameter file.
    recover : `bool`, optional (default=`False`)
        If `True`, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see `Model.from_archive`.
    force : `bool`, optional (default=`False`)
        If `True`, we will overwrite the serialization directory if it already exists.
    node_rank : `int`, optional
        Rank of the current node in distributed training
    include_package : `str`, optional
        In distributed mode, extra packages mentioned will be imported in trainer workers.
    dry_run : `bool`, optional (default=`False`)
        Do not train a model, but create a vocabulary, show dataset statistics and other training
        information.
    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    # Returns
    best_model : `Optional[Model]`
        The model with the best epoch weights or `None` if in dry run.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    if parameter_filename.endswith(('.yaml', '.yml')):
        params = yaml_to_params(parameter_filename, overrides)
    else:
        params = Params.from_file(parameter_filename, overrides)

    if not serialization_dir:
        config_dir = os.path.dirname(parameter_filename)
        serialization_dir = os.path.join(config_dir, 'serialization')

    return train_model(params=params,
                       serialization_dir=serialization_dir,
                       recover=recover,
                       force=force,
                       node_rank=node_rank,
                       include_package=include_package,
                       dry_run=dry_run,
                       file_friendly_logging=file_friendly_logging)


def yaml_to_params(params_file: str, overrides: str = "") -> Params:
    # redirect to cache, if necessary
    params_file = cached_path(params_file)

    with open(params_file) as f:
        file_dict = yaml.safe_load(f)

    overrides_dict = parse_overrides(overrides)
    param_dict = with_fallback(preferred=overrides_dict, fallback=file_dict)

    return Params(param_dict)
