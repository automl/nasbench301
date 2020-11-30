import glob
import json
import os
import shutil
import time

import click
import matplotlib

from nasbench301.surrogate_models import utils

matplotlib.use('Agg')


@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
@click.option('--data_config_path', type=click.STRING, help='Path to config.json',
              default='surrogate_models/configs/data_configs/nb_301.json')
@click.option('--left_out_optimizer', type=click.STRING, help='Optimizer that is left out in data_config',
              default='None')
@click.option('--splits_log_dir', type=click.STRING, help='Experiment directory',
              default='surrogate_models/configs/data_splits/default_split')
@click.option('--seed', type=click.INT, help='seed for numpy, python, pytorch', default=6)
def create_data_splits(nasbench_data, data_config_path, left_out_optimizer, splits_log_dir, seed):
    # Load config
    model = "lgb"
    model_config_path = None
    data_config = json.load(open(data_config_path, 'r'))

    # Create log directory
    log_dir = os.path.join("tmp", '{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), seed))
    os.makedirs(log_dir)
    os.makedirs(splits_log_dir, exist_ok=True)

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
    print("==> Instantiating surrogate")
    surrogate_model = utils.model_dict[model](data_root=nasbench_data,
                                              log_dir=log_dir,
                                              seed=seed,
                                              model_config=model_config,
                                              data_config=data_config)

    if left_out_optimizer != 'None':
        if len(surrogate_model.test_paths) > 0:
            raise ValueError("Data config contains test split but test config paths were specified")
        if left_out_optimizer not in ['darts', 'bananas', 'combo', 'de', 're', 'tpe', 'random_ws', 'pc_darts', 'gdas',
                                      'drnas']:
            raise ValueError(
                "Optimizer has to be in ['darts', 'bananas', "
                "'combo', 'de', 're', 'tpe', 'random_ws', 'pc_darts', 'gdas', 'drnas]")
        print("==> Loading %s configs as test split" % left_out_optimizer)
        test_paths = glob.glob(os.path.join(nasbench_data, left_out_optimizer, '*'))
        surrogate_model.test_paths = test_paths

    # Save data splits
    print("==> Saving data splits")
    json.dump(surrogate_model.train_paths, open(os.path.join(splits_log_dir, "train_paths.json"), "w"))
    json.dump(surrogate_model.val_paths, open(os.path.join(splits_log_dir, "val_paths.json"), "w"))
    json.dump(surrogate_model.test_paths, open(os.path.join(splits_log_dir, "test_paths.json"), "w"))

    shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    create_data_splits()
