import json
import os
import time

import click
import matplotlib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from nasbench301.surrogate_models import utils
from nasbench301.surrogate_models.ensemble import Ensemble

matplotlib.use('Agg')


@click.command()
@click.option('--model', type=click.Choice(list(utils.model_dict.keys())), default='gnn',
              help='which surrogate model to fit')
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
@click.option('--model_config_path', type=click.STRING, help='Leave None to use the default config.', default=None)
@click.option('--data_config_path', type=click.STRING, help='Path to config.json',
              default='surrogate_models/configs/data_configs/nb_301.json')
@click.option('--log_dir', type=click.STRING, help='Experiment directory', default='experiments/surrogate_models')
@click.option('--seed', type=click.INT, help='seed for numpy, python, pytorch', default=6)
@click.option('--ensemble', help='wether to use an ensemble', default=False)
@click.option('--data_splits_root', type=click.STRING, help='path to directory containing data splits', default=None)
def train_surrogate_model(model, nasbench_data, model_config_path, data_config_path, log_dir, seed, ensemble,
                          data_splits_root):
    # Load config
    data_config = json.load(open(data_config_path, 'r'))

    # Create log directory
    log_dir = os.path.join(log_dir, model, '{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), seed))
    os.makedirs(log_dir)

    # Select model config to use
    if model_config_path is None:
        # Get model configspace
        model_configspace = utils.get_model_configspace(model)

        # Use default model config
        model_config = model_configspace.get_default_configuration().get_dictionary()
    else:
        model_config = json.load(open(model_config_path, 'r'))
    model_config['model'] = model

    # Instantiate surrogate model
    if ensemble:
        surrogate_model = Ensemble(member_model_name=model, data_root=nasbench_data, log_dir=log_dir,
                                   starting_seed=seed,
                                   model_config=model_config, data_config=data_config, ensemble_size=5)
    else:
        surrogate_model = utils.model_dict[model](data_root=nasbench_data, log_dir=log_dir, seed=seed,
                                                  model_config=model_config, data_config=data_config)

    # Override train/val/test splits if specified
    if data_splits_root is not None:
        train_paths = json.load(open(os.path.join(data_splits_root, "train_paths.json"), "r"))
        val_paths = json.load(open(os.path.join(data_splits_root, "val_paths.json"), "r"))
        test_paths = json.load(open(os.path.join(data_splits_root, "test_paths.json"), "r"))

        cross_val_paths = train_paths + val_paths
        optimzier_identifiers = [path.split("/")[-2] for path in cross_val_paths]
        k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=6)
        splits = list(k_fold.split(cross_val_paths, optimzier_identifiers))

        train_inds, val_inds = splits[seed % len(splits)]

        surrogate_model.train_paths = list(np.array(cross_val_paths)[train_inds])
        surrogate_model.val_paths = list(np.array(cross_val_paths)[val_inds])
        surrogate_model.test_paths = test_paths

    # Train and validate the model on the available data
    surrogate_model.train()

    # Test the model
    if len(surrogate_model.test_paths) > 0:
        surrogate_model.test()

    # Save the model
    surrogate_model.save()


if __name__ == "__main__":
    train_surrogate_model()
