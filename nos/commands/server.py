import logging
import os

import torch
from allennlp.common.util import prepare_environment
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from jina.flow import Flow

from .train import yaml_to_params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def serve_from_file(archive_path, model_path, flow_path, overrides=None, device=0):
    # if archive_path.endswith('gz'):
    #     archive = load_archive(archive_path, device, overrides)
    #     config = archive.config
    #     prepare_environment(config)
    #     model = archive.model
    #     serialization_dir = os.path.dirname(archive_path)
    # elif archive_path.endswith('yaml'):
    #     config = yaml_to_params(archive_path, overrides)
    #     prepare_environment(config)
    #     config_dir = os.path.dirname(archive_path)
    #     serialization_dir = os.path.join(config_dir, 'serialization')

    # os.makedirs(serialization_dir, exist_ok=True)
    # vocab = Vocabulary.from_params(config.pop('vocabulary'))

    # model = Model.from_params(vocab=vocab, params=config.pop('model'))
    # if torch.cuda.is_available():
    #     device = torch.device(f'cuda:{device}')
    # else:
    #     device = torch.device('cpu')

    # if model_path:
    #     best_model_state = torch.load(model_path, device)
    #     model.load_state_dict(best_model_state)

    # We want to keep the dropout to generate confidence intervals
    # model = model.eval()

    print('init flow')
    flow = Flow()
    flow.add(name='radflow_forecaster', uses=flow_path)

    print('start loop')
    with flow:
        flow.block()
