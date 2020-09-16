import glob
import json
import os

import click
import matplotlib
import numpy as np
import tqdm

matplotlib.use('Agg')


def unique_configs(paths_to_json):
    """
    Checks to configurations in the results.json files and returns paths such that none contains
    duplicate configurations.
    :param paths_to_json: List of dir/results.json
    :return: unique list of dir/results.json w.r.t. configuration
    """
    config_hashes = []

    for path_to_json in paths_to_json:
        with open(path_to_json, "r") as f:
            results = json.load(f)
        config_hash = hash(results["optimized_hyperparamater_config"].__repr__())
        config_hashes.append(config_hash)

    _, unique_indices = np.unique(config_hashes, return_index=True)

    return list(np.array(paths_to_json)[unique_indices])


@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
def create_data_splits(nasbench_data):
    all_paths = glob.glob(os.path.join(nasbench_data, '*', '*.json'))
    good_config_paths, bad_config_paths = [], []
    for path in tqdm.tqdm(all_paths, desc='Reading dataset'):
        config = json.load(open(path, 'r'))
        if config['info'][0]['val_accuracy'] < 92:
            bad_config_paths.append(path)
        else:
            good_config_paths.append(path)

    good_config_paths = unique_configs(good_config_paths)
    bad_config_paths = unique_configs(bad_config_paths)

    # Randomly shuffle the list
    rng = np.random.RandomState(6)
    rng.shuffle(good_config_paths)

    # Extract the train/val/test splits
    train_ratio, val_ratio = 0.9, 0.1
    train_upper_idx = int(train_ratio * len(good_config_paths))
    val_upper_idx = int((train_ratio + val_ratio) * len(good_config_paths))

    train_paths = good_config_paths[:train_upper_idx]
    val_paths = good_config_paths[train_upper_idx:val_upper_idx]
    test_paths = bad_config_paths

    save_path = 'surrogate_models/configs/data_splits/good_region_ablation_study'
    os.makedirs(save_path, exist_ok=True)

    json.dump(train_paths, open(os.path.join(save_path, 'train_paths.json'), 'w'))
    json.dump(val_paths, open(os.path.join(save_path, 'val_paths.json'), 'w'))
    json.dump(test_paths, open(os.path.join(save_path, 'test_paths.json'), 'w'))


if __name__ == "__main__":
    create_data_splits()
