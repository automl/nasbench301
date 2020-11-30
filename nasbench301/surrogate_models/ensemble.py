import json
import logging
import os
import sys
import numpy as np

from abc import abstractmethod
from nasbench301.surrogate_models import utils


class AbstractEnsemble():
    def __init__(self, member_model_name, data_root, log_dir, starting_seed, model_config, data_config, ensemble_size):
        self.member_model_name = member_model_name
        self.member_model_init_dict = {"data_root":data_root, "model_config":model_config, "data_config":data_config}
        self.base_logdir = log_dir
        self.ensemble_size = ensemble_size
        self.starting_seed = starting_seed
        self.model_config = model_config
        self.data_config = data_config

        self.train_paths = None
        self.val_paths = None
        self.test_paths = None

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self, model_paths):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def query(self, config_dict):
        raise NotImplementedError

    @abstractmethod
    def query_with_noise(self, config_dict):
        raise NotImplementedError


class BaggingEnsemble(AbstractEnsemble):
    def __init__(self, member_model_name, data_root, log_dir, starting_seed, model_config, data_config, ensemble_size, init_ensemble=True):
        super(BaggingEnsemble, self).__init__(member_model_name, data_root, log_dir, starting_seed, model_config, data_config, ensemble_size)

        if init_ensemble:
            self.init_ensemble()

    def init_ensemble(self):
        """Initializes the ensemble members and their logdirs"""
        self.member_logdirs = []
        self.ensemble_members = []
        for ind in range(self.ensemble_size):
            member_logdir = os.path.join(self.base_logdir, "ensemble_member_"+str(ind))
            
            ensemble_member = utils.model_dict[self.member_model_name](log_dir=member_logdir, seed=self.starting_seed+ind, **self.member_model_init_dict)
            
            if self.train_paths == None:
                self.train_paths = ensemble_member.train_paths
                self.val_paths = ensemble_member.val_paths
                self.test_paths = ensemble_member.test_paths

            ensemble_member.train_paths = self.train_paths
            ensemble_member.val_paths = self.val_paths
            ensemble_member.test_paths = self.test_paths

            self.ensemble_members.append(ensemble_member)
            self.member_logdirs.append(member_logdir)

    def save(self):
        """Save all ensembles and the data/model configs to self.base_logdir"""
        # Log data and model config to basedir
        data_config_logdir = os.path.join(self.base_logdir, "data_config.json")
        model_config_logdir = os.path.join(self.base_logdir, "model_config.json")

        with open(data_config_logdir, "w") as f:
            json.dump(self.data_config, f)

        with open(model_config_logdir, "w") as f:
            json.dump(self.model_config, f)

        # Log ensemble members
        for ens_mem in self.ensemble_members:
            ens_mem.save()

        # Log train/val/test paths
        json.dump(self.train_paths, open(os.path.join(self.base_logdir, "train_paths.json"), "w"))
        json.dump(self.val_paths, open(os.path.join(self.base_logdir, "val_paths.json"), "w"))
        json.dump(self.test_paths, open(os.path.join(self.base_logdir, "test_paths.json"), "w"))

    def load(self, model_paths=None, train_paths=None, val_paths=None, test_paths=None):
        """Loads all models in model_paths to the ensemble member list. If not specified uses its own self.member_logdirs"""
        # Load data splits
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        
        # Get member directories
        if isinstance(model_paths, list):
            self.member_logdirs = model_paths
        if isinstance(model_paths, str) and model_paths.endswith('.model'): # compatibility with loading from /surrogate_model.model
            self.member_logdirs = [p for p in os.listdir(os.path.dirname(model_paths)) if os.path.isdir(p)]
        
        # Load ensemble members
        self.ensemble_members = []
        for ind, member_logdir in enumerate(self.member_logdirs):
            ens_mem = utils.model_dict[self.member_model_name](log_dir=None, seed=self.starting_seed+ind, **self.member_model_init_dict)
            ens_mem.load(os.path.join(member_logdir, 'surrogate_model.model'))
            ens_mem.train_paths = train_paths
            ens_mem.val_paths = val_paths
            ens_mem.test_paths = test_paths
            ens_mem.log_dir = member_logdir
            self.ensemble_members.append(ens_mem)

    def add_member(self, surrogate_model):
        """Add a surrogate model to the bagging ensemble"""
        surrogate_model.train_paths = self.train_paths
        surrogate_model.val_paths = self.val_paths
        surrogate_model.test_paths = self.test_paths
        self.ensemble_members.append(surrogate_model)
        self.member_logdirs.append(surrogate_model.log_dir)
        
    def train(self):
        """Train the ensemble members"""
        for ens_mem in self.ensemble_members:
            ens_mem.train()

    def validate(self):
        """Call .validate on the ensemble members"""
        for ens_mem in self.ensemble_members:
            ens_mem.validate()

    def test(self):
        """Call .test on the ensemble members"""
        for ens_mem in self.ensemble_members:
            ens_mem.test()

    def validate_ensemble(self, apply_noise):
        """Get validation metrics using ensemble predictions"""
        val_paths = self.get_val_paths()
        metrics, preds, stddevs, targets = self.evaluate_ensemble(val_paths, apply_noise)
        
        logging.info('==> Ensemble validation metrics %s', metrics)
        return metrics, preds, stddevs, targets

    def test_ensemble(self, apply_noise):
        """Get test metrics using ensemble predictions"""
        test_paths = self.get_test_paths()
        metrics, preds, stddevs, targets = self.evaluate_ensemble(test_paths, apply_noise)

        logging.info('==> Ensemble test metrics %s', metrics)
        return metrics, preds, stddevs, targets

    def evaluate_ensemble(self, result_paths, apply_noise):
        """Evaluates the metrics on the result paths using ensemble predicitons"""
        preds, targets = [], []

        # Collect individuals predictions
        for member_model in self.ensemble_members:
            member_metrics, member_preds, member_targets = member_model.evaluate(result_paths)
            logging.info("==> Eval member metrics: %s", member_metrics)
            if len(targets)==0:
                preds.append(member_preds)
                targets = member_targets
                continue
            if np.any((targets-member_targets)>1e-5):
                raise ValueError("Ensemble members have different targets!")
            preds.append(member_preds)

        means = np.mean(preds, axis=0)
        stddevs = np.std(preds, axis=0)

        # Apply noise
        if apply_noise:
            noisy_predictions = [np.random.normal(loc=mean, scale=stddev, size=1)[0] for mean, stddev in zip(means, stddevs)]
            ensemble_predictions = noisy_predictions
        else:
            ensemble_predictions = means

        # Evaluate metrics
        metrics = utils.evaluate_metrics(targets, ensemble_predictions, prediction_is_first_arg=False)

        return metrics, ensemble_predictions, stddevs, targets

    def get_train_paths(self):
        """
        all_member_train_paths = [member_model.train_paths for member_model in self.ensemble_members]
        for member_train_paths in all_member_train_paths:
            if member_train_paths==all_member_train_paths[0]:
                continue
            else:
                raise ValueError("Ensemble member have different data splits!")
        return all_member_train_paths[0]
        """
        return self.train_paths

    def get_val_paths(self):
        """
        all_member_val_paths = [member_model.val_paths for member_model in self.ensemble_members]
        for member_val_paths in all_member_val_paths:
            if member_val_paths==all_member_val_paths[0]:
                continue
            else:
                raise ValueError("Ensemble member have different data splits!")
        return all_member_val_paths[0]
        """
        return self.val_paths

    def get_test_paths(self):
        """
        all_member_test_paths = [member_model.test_paths for member_model in self.ensemble_members]
        for member_test_paths in all_member_test_paths:
            if member_test_paths==all_member_test_paths[0]:
                continue
            else:
                raise ValueError("Ensemble member have different data splits!")
        return all_member_test_paths[0]
        """
        return self.test_paths

    def log_dataset_predictions(self):
        """Log predictions for all members"""
        for ens_mem in self.ensemble_members:
            ens_mem.log_dataset_predictions()

    def query_members(self, config_dict):
        """Get predictions from the ensemble members"""
        ensemble_preds = []
        for ens_mem in self.ensemble_members:
            ensemble_preds.append(ens_mem.query(config_dict))
        return ensemble_preds

    def query_mean(self, config_dict):
        """Get the mean of the ensemble member predictions"""
        ensemble_preds = self.query_members(config_dict)
        return np.mean(ensemble_preds)

    def query_std(self, config_dict):
        """Get the stddev of the ensemble member predictions"""
        ensemble_preds = self.query_members(config_dict)
        return np.std(ensemble_preds)

    def query(self, config_dict):
        """Query the ensemble with noise"""
        ensemble_preds = self.query_members(config_dict)
        noise = np.random.normal(0, np.std(ensemble_preds), 1)[0]
        return np.mean(ensemble_preds)+noise


class Ensemble():
    
    def __new__(cls, member_model_name, data_root, log_dir, starting_seed, model_config, data_config, ensemble_size=10, init_ensemble=True):
        
        if member_model_name=="ngb":
            raise NotImplementedError("Ensembles not implemented for ngb!")

        if "forest" in member_model_name:    # random forests have their own query_with_noise implementation
            return utils.model_dict[member_model_name](data_root, log_dir, starting_seed, model_config, data_config)
        
        else:                                # use bagging ensembles for everything else
            return BaggingEnsemble(member_model_name, data_root, log_dir, starting_seed, model_config, data_config, ensemble_size, init_ensemble)
