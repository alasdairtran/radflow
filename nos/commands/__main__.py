"""Train and run semantic diff models.

Usage:
    nos (train|generate|evaluate) [options] PARAM_PATH
    nos (-h | --help)
    nos (-v | --version)

Options:
    -e --expt-dir EXPT_PATH
                        Directory to store experiment results and model files.
                        If not given, they will be stored in the same directory
                        as the parameter file.
    -r, --recover       Recover training from existing model.
    -f, --force    Delete existing models and logs.
    -o --overrides OVERRIDES
                        A JSON structure used to override the experiment
                        configuration.
    -u --pudb           Enable debug mode with pudb.
    -p --ptvsd PORT     Enable debug mode with ptvsd on a given port, for
                        example 5678.
    -g --file-friendly-logging
                        Outputs tqdm status on separate lines and slows tqdm
                        refresh rate
    -i --include-package PACKAGE
                        Additional packages to include.
    -q --quiet          Print less info
    -s --eval-suffix S  Evaluation generation file name [default: ]
    PARAM_PATH          Path to file describing the model parameters.
    -m --model-path PATH Path the the best model.
    -d --with-dropout   Evaluate with dropout on.

Examples:
    nos train -r -g expt/lstm/config.yaml
"""

import logging
import os

import ptvsd
import pudb
from docopt import docopt
from schema import And, Or, Schema, Use

from nos.commands.evaluate import evaluate_from_file
from nos.commands.train import train_model_from_file
from nos.utils import setup_logger

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'param_path': Or(None, os.path.exists),
        'model_path': Or(None, os.path.exists),
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'eval_suffix': str,
        object: object,
    })
    args = schema.validate(args)
    args['debug'] = args['ptvsd'] or args['pudb']
    return args


def main():
    """Parse command line arguments and execute script."""
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['debug']:
        logger.setLevel(logging.DEBUG)
    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()
    elif args['pudb']:
        pudb.set_trace()

    if args['quiet']:
        # Disable some of the more verbose logging statements
        logging.getLogger('allennlp.common.params').disabled = True
        logging.getLogger('allennlp.nn.initializers').disabled = True
        logging.getLogger(
            'allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    if args['train']:
        train_model_from_file(
            parameter_filename=args['param_path'],
            serialization_dir=args['expt_dir'],
            overrides=args['overrides'],
            file_friendly_logging=args['file_friendly_logging'],
            recover=args['recover'],
            force=args['force'])

    elif args['evaluate']:
        evaluate_from_file(args['param_path'], args['model_path'],
                           args['overrides'], args['eval_suffix'],
                           args['with_dropout'])


if __name__ == '__main__':
    main()
