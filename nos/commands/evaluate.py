import json
import logging
import os
from typing import Any, Dict, Iterable

import numpy as np
import torch
from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data import DataLoader, Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util
from allennlp.training.util import HasBeenWarned, datasets_from_params

from .train import yaml_to_params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def evaluate_from_file(archive_path, model_path, overrides=None, eval_suffix='', device=0):
    if archive_path.endswith('gz'):
        archive = load_archive(archive_path, device, overrides)
        config = archive.config
        prepare_environment(config)
        model = archive.model
        serialization_dir = os.path.dirname(archive_path)
    elif archive_path.endswith('yaml'):
        config = yaml_to_params(archive_path, overrides)
        prepare_environment(config)
        config_dir = os.path.dirname(archive_path)
        serialization_dir = os.path.join(config_dir, 'serialization')

    os.makedirs(serialization_dir, exist_ok=True)
    all_datasets = datasets_from_params(config)

    # We want to create the vocab from scratch since it might be of a
    # different type. Vocabulary.from_files will always create the base
    # Vocabulary instance.
    # if os.path.exists(os.path.join(serialization_dir, "vocabulary")):
    #     vocab_path = os.path.join(serialization_dir, "vocabulary")
    #     vocab = Vocabulary.from_files(vocab_path)
    vocab = Vocabulary.from_params(config.pop('vocabulary'))

    model = Model.from_params(vocab=vocab, params=config.pop('model'))

    if model_path:
        best_model_state = torch.load(model_path)
        model.load_state_dict(best_model_state)

    instances = all_datasets.get('test')
    data_loader_params = config.pop("validation_data_loader")
    data_loader = DataLoader.from_params(
        dataset=instances, params=data_loader_params)

    model.eval().to(device)
    model.evaluate_mode = True

    metrics = evaluate(model, data_loader,
                       device, serialization_dir, eval_suffix, batch_weight_key='')

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        if isinstance(metric, list):
            if key not in ['smapes']:
                continue
            metric_array = np.array(metric)
            logger.info(f"{key}_min: {np.amin(metric_array)}")
            logger.info(f"{key}_q1: {np.quantile(metric_array, 0.25)}")
            logger.info(f"{key}_median: {np.median(metric_array)}")
            logger.info(f"{key}_mean: {np.mean(metric_array)}")
            logger.info(f"{key}_q3: {np.quantile(metric_array, 0.75)}")
            logger.info(f"{key}_max: {np.amax(metric_array)}")
        else:
            logger.info("%s: %s", key, metric)

    output_file = os.path.join(
        serialization_dir, f"evaluate-metrics{eval_suffix}.json")
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics


def evaluate(model: Model,
             data_loader: DataLoader,
             cuda_device: int,
             serialization_dir: str,
             eval_suffix: str,
             batch_weight_key: str) -> Dict[str, Any]:
    check_for_gpu(cuda_device)

    with torch.no_grad():
        model.eval()

        iterator = iter(data_loader)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator)

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        smape = []
        smapes_all = []
        daily_errors = []
        preds = []
        preds_all = []
        keys = []
        f_parts = []
        for batch in generator_tqdm:
            batch_count += 1
            batch = nn_util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            loss = output_dict.get("loss")
            smape += output_dict['smapes']
            daily_errors += output_dict['daily_errors']
            keys += output_dict['keys']
            if 'preds' in output_dict:
                preds += output_dict['preds']

            if 'f_parts' in output_dict:
                f_parts += output_dict['f_parts']

            if 'smapes_all' in output_dict:
                smapes_all += output_dict['smapes_all']

            if 'preds_all' in output_dict:
                preds_all += output_dict['preds_all']

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if (not HasBeenWarned.tqdm_ignores_underscores and
                    any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                HasBeenWarned.tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        final_metrics['smapes'] = smape
        final_metrics['smapes_all'] = smapes_all
        final_metrics['daily_errors'] = daily_errors
        final_metrics['preds'] = preds
        final_metrics['preds_all'] = preds_all
        final_metrics['f_parts'] = f_parts

        keys = [int(k) for k in keys]
        final_metrics['keys'] = keys
        if loss_count > 0:
            # Sanity check
            # if loss_count != batch_count:
            #     raise RuntimeError("The model you are trying to evaluate only sometimes " +
            #                        "produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

    return final_metrics
