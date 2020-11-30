import os
import json
from pathlib import Path

from nasbench301 import representations
from nasbench301.surrogate_models import utils
from nasbench301.surrogate_models.ensemble import Ensemble


fixed_hyperparameters = {
        "CreateImageDataLoader:batch_size": 96,
        "ImageAugmentation:augment": "True",
        "ImageAugmentation:cutout": "True",
        "ImageAugmentation:cutout_holes": 1,
        "ImageAugmentation:cutout_length": 16,
        "ImageAugmentation:autoaugment": "False",
        "ImageAugmentation:fastautoaugment": "False",
        "LossModuleSelectorIndices:loss_module": "cross_entropy",
        "NetworkSelectorDatasetInfo:darts:auxiliary": "True",
        "NetworkSelectorDatasetInfo:darts:drop_path_prob": 0.2,
        "NetworkSelectorDatasetInfo:network": "darts",
        "OptimizerSelector:optimizer": "sgd",
        "OptimizerSelector:sgd:learning_rate": 0.025,
        "OptimizerSelector:sgd:momentum": 0.9,
        "OptimizerSelector:sgd:weight_decay": 0.0003,
        "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": 100,
        "SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min": 1e-8,
        "SimpleLearningrateSchedulerSelector:lr_scheduler": "cosine_annealing",
        "SimpleTrainNode:batch_loss_computation_technique": "mixup",
        "SimpleTrainNode:mixup:alpha": 0.2,
        "NetworkSelectorDatasetInfo:darts:init_channels": 32,
        "NetworkSelectorDatasetInfo:darts:layers": 8
        }


def load_ensemble(ensemble_parent_directory):
    """Loads a surrogate ensemble
    
    args:
        ensemble_parent_directory: directory which contains the ensemble members. Members must be the same model type
    """

    ensemble_member_dirs = [os.path.dirname(filename) for filename in Path(ensemble_parent_directory).rglob('*surrogate_model.model')]
    data_config = json.load(open(os.path.join(ensemble_member_dirs[0], 'data_config.json'), 'r'))
    model_config = json.load(open(os.path.join(ensemble_member_dirs[0], 'model_config.json'), 'r'))

    surrogate_model = Ensemble(member_model_name=model_config['model'],
                               data_root='None', 
                               log_dir=ensemble_parent_directory,
                               starting_seed=data_config["seed"],
                               model_config=model_config,
                               data_config=data_config, 
                               ensemble_size=len(ensemble_member_dirs),
                               init_ensemble=False)

    surrogate_model.load(model_paths=ensemble_member_dirs)

    surrogate_api = SurrogateAPI(surrogate_model)

    return surrogate_api


class SurrogateAPI():
    """Wrapper for a surrogate ensemble"""

    def __init__(self, surrogate_model):
        """
        args:
            surrogate_model: An instance of Ensemble
        """
        self.model = surrogate_model
        self.representations_converters = representations.CONVERTER_DICT

    def convert_representation(self, config, representation):
        """Convert representation to a dictionary"""
        if not representation in self.representations_converters.keys():
            raise NotImplementedError("%s representation is not supported, please choose from %s" %(representation, self.representations_converters.keys()))
        
        converter = self.representations_converters[representation]()
        config_dict = converter.convert(config)
        return config_dict

    def predict(self, config, representation, with_noise=True):
        """Return the mean over the predictions of surrogate ensemble individuals with or without noise
        
        args:
            config: An architecture to query, given in any of the supported representations
            representation: str, representation used for the config 
            with_noise: bool, wether to apply noise or only use the mean of the ensemble members as prediction
        """
        config_dict = self.convert_representation(config, representation)

        config_dict = {**fixed_hyperparameters, **config_dict}

        if with_noise:
            pred = self.model.query(config_dict)
        else:
            pred = self.model.query_mean(config_dict)
        return pred
