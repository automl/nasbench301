import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from IPython import embed

import numpy as np
import pathvalidate
import torch
import torch.backends.cudnn as cudnn

from nasbench301.surrogate_models import utils


class SurrogateModel(ABC):
    def __init__(self, data_root, log_dir, seed, model_config, data_config):
        self.data_root = data_root
        self.log_dir = log_dir
        self.model_config = model_config
        self.data_config = data_config
        self.seed = seed

        # Seeding
        np.random.seed(seed)
        cudnn.benchmark = True
        torch.manual_seed(seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(seed)

        # NOTE: Update to use absolute path, also moved configspace to
        #       be included in the installed package
        current_dir = os.path.dirname(os.path.abspath(__file__))
        nasbench_root = os.path.join(current_dir, os.pardir)
        configspace_path = os.path.join(nasbench_root, 'configspace.json')

        # Create config loader
        self.config_loader = utils.ConfigLoader(configspace_path)

        # Load the data
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            # Add logger
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)

            # Dump the config of the run to log_dir
            self.data_config['seed'] = seed

            logging.info('MODEL CONFIG: {}'.format(model_config))
            logging.info('DATA CONFIG: {}'.format(data_config))
            self._load_data()
            logging.info(
                'DATA: No. train data {}, No. val data {}, No. test data {}'.format(len(self.train_paths),
                                                                                    len(self.val_paths),
                                                                                    len(self.test_paths)))
            with open(os.path.join(log_dir, 'model_config.json'), 'w') as fp:
                json.dump(model_config, fp)

            with open(os.path.join(log_dir, 'data_config.json'), 'w') as fp:
                json.dump(data_config, fp)

            #with open(os.path.join(log_dir, 'train_paths.json'), 'w') as fp:
            #    json.dump(self.train_paths, fp)

            #with open(os.path.join(log_dir, 'val_paths.json'), 'w') as fp:
            #    json.dump(self.val_paths, fp)

            #with open(os.path.join(log_dir, 'test_paths.json'), 'w') as fp:
            #    json.dump(self.test_paths, fp)

    def _load_data(self):
        # Get the result train/val/test split
        train_paths = []
        val_paths = []
        test_paths = []
        for key, data_config in self.data_config.items():
            if type(data_config) == dict:
                result_loader = utils.ResultLoader(
                    self.data_root, filepath_regex=data_config['filepath_regex'],
                    train_val_test_split=data_config, seed=self.seed)
                train_val_test_split = result_loader.return_train_val_test()
                # Save the paths
                for paths, filename in zip(train_val_test_split, ['train_paths', 'val_paths', 'test_paths']):
                    file_path = os.path.join(self.log_dir,
                                             pathvalidate.sanitize_filename('{}_{}.json'.format(key, filename)))
                    json.dump(paths, open(file_path, 'w'))

                train_paths.extend(train_val_test_split[0])
                val_paths.extend(train_val_test_split[1])
                test_paths.extend(train_val_test_split[2])

        '''
        # Add extra paths to test
        # Increased ratio of skip-connections.
        matching_files = lambda dir: [str(path) for path in Path(os.path.join(self.data_root, dir)).rglob('*.json')]
        test_paths.extend(matching_files('groundtruths/low_parameter/'))

        # Extreme hyperparameter settings
        # Learning rate
        test_paths.extend(matching_files('groundtruths/hyperparameters/learning_rate/'))
        test_paths.extend(matching_files('groundtruths/hyperparameters/weight_decay/'))

        # Load the blacklist to filter out those elements
        if self.model_config["model"].endswith("_time"):
            blacklist = json.load(open('surrogate_models/configs/data_configs/blacklist_runtimes.json'))
        else:
            blacklist = json.load(open('surrogate_models/configs/data_configs/blacklist.json'))
        filter_out_black_list = lambda paths: list(filter(lambda path: path not in blacklist, paths))
        train_paths, val_paths, test_paths = map(filter_out_black_list, [train_paths, val_paths, test_paths])
        '''
        # Shuffle the total file paths again
        rng = np.random.RandomState(6)
        rng.shuffle(train_paths)
        rng.shuffle(val_paths)
        rng.shuffle(test_paths)

        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths

    def _get_labels_and_preds(self, result_paths):
        """Get labels and predictions from json paths"""
        labels = []
        preds = []
        for result_path in result_paths:
            config_space_instance, val_accuracy_true, test_accuracy_true, _ = self.config_loader[result_path]
            val_pred = self.query(config_space_instance.get_dictionary())
            labels.append(val_accuracy_true)
            preds.append(val_pred)

        return labels, preds

    def _log_predictions(self, result_paths, labels, preds, identifier):
        """Log paths, labels and predictions for one split"""
        if not isinstance(preds[0], float):
            preds = [p[0] for p in preds]

        logdir = os.path.join(self.log_dir, identifier+"_preds.json")
        
        dump_dict = {"paths": result_paths, "labels": labels, "predictions": preds}
        with open(logdir, "w") as f:
            json.dump(dump_dict, f)

    def log_dataset_predictions(self):
        """Log paths, labels and predictions for train, val, test splits"""
        data_splits = {"train": self.train_paths, "val": self.val_paths, "test": self.test_paths}

        for split_identifier, result_paths in data_splits.items():
            print("==> Logging predictions of %s split" %split_identifier)
            labels, preds = self._get_labels_and_preds(result_paths)
            self._log_predictions(result_paths, labels, preds, split_identifier)

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def validate(self):
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        raise NotImplementedError()

    @abstractmethod
    def load(self, model_path):
        raise NotImplementedError()

    @abstractmethod
    def query(self, config_dict):
        raise NotImplementedError()
