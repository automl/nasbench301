import re
from functools import partial, wraps
from random import shuffle

import ConfigSpace
import numpy as np
import torch
from torch.utils.data import Dataset

from nasbench301.surrogate_models.bananas.bananas_src.darts.arch import Arch
from nasbench301.surrogate_models.bananas.darts_src.genotypes import Genotype

OPS = ['max_pool_3x3',
       'avg_pool_3x3',
       'skip_connect',
       'sep_conv_3x3',
       'sep_conv_5x5',
       'dil_conv_3x3',
       'dil_conv_5x5']


def create_genotype(func):
    @wraps(func)
    def genotype_wrapper(*args, **kwargs):
        normal = func(*args, cell_type='normal', **kwargs)
        reduction = func(*args, cell_type='reduce', **kwargs)
        concat = list(range(2, 6))
        return Genotype(normal, concat, reduction, concat)

    return genotype_wrapper


only_numeric_fn = lambda x: int(re.sub("[^0-9]", "", x))
custom_sorted = partial(sorted, key=only_numeric_fn)


class BANANASDataset(Dataset):
    def __init__(self, result_paths, config_loader):
        super(BANANASDataset, self).__init__()
        self.result_paths = result_paths
        shuffle(self.result_paths)
        self.config_loader = config_loader

    def __len__(self):
        return len(self.result_paths)

    def __getitem__(self, idx):
        config_space_instance, val_accuracy, test_accuracy, _ = self.config_loader[self.result_paths[idx]]
        bananas_format = self.convert_config_space_dict_to_bananas_format(
            config_space_instance.get_dictionary())
        enc = Arch(bananas_format).encode_paths()
        return enc, val_accuracy

    def convert_to_bananas_paths_format(self, config_space_instance):
        bananas_format = self.convert_config_space_dict_to_bananas_format(
            config_space_instance.get_dictionary())
        enc = Arch(bananas_format).encode_paths()
        return enc
    
    def convert_config_space_dict_to_bananas_format(self, config):
        genotype = self.parse_config(config, config_space=self.config_loader.config_space)
        normal = [(node, np.where(np.array(OPS) == op)[0][0]) for op, node in genotype.normal]
        reduction = [(node, np.where(np.array(OPS) == op)[0][0]) for op, node in genotype.reduce]
        bananas_arch = (normal, reduction)
        return bananas_arch

    @create_genotype
    def parse_config(self, config, config_space, cell_type):
        cell = []
        config = ConfigSpace.Configuration(config_space, config)
        edges = custom_sorted(
            list(
                filter(
                    re.compile('.*edge_{}*.'.format(cell_type)).match,
                    config_space.get_active_hyperparameters(config)
                )
            )
        ).__iter__()
        nodes = custom_sorted(
            list(
                filter(
                    re.compile('.*inputs_node_{}*.'.format(cell_type)).match,
                    config_space.get_active_hyperparameters(config)
                )
            )
        ).__iter__()
        op_1 = config[next(edges)]
        op_2 = config[next(edges)]
        cell.extend([(op_1, 0), (op_2, 1)])
        for node in nodes:
            op_1 = config[next(edges)]
            op_2 = config[next(edges)]
            input_1, input_2 = map(int, config[node].split('_'))
            cell.extend([(op_1, input_1), (op_2, input_2)])
        return cell

    def convert_bananas_to_config_space_dict(self, bananas_arch):
        """
        BANANAS representation is like the DARTS genotype, but the tuples are inverted (first node, then operation).
        """
        # Convert to DARTS Genotype
        normal = [(OPS[op], node) for node, op in bananas_arch[0]]
        reduction = [(OPS[op], node) for node, op in bananas_arch[1]]
        concat = list(range(2, 6))
        genotype = Genotype(normal=normal, reduce=reduction, normal_concat=concat, reduce_concat=concat)
        config = self.convert_genotype_to_config(genotype)
        return config

    def convert_genotype_to_config(self, arch):
        base_string = 'NetworkSelectorDatasetInfo:darts:'
        config = {}

        for cell_type in ['normal', 'reduce']:
            cell = eval('arch.' + cell_type)

            start = 0
            n = 2
            for node_idx in range(4):
                end = start + n
                ops = cell[2 * node_idx: 2 * node_idx + 2]

                # get edge idx
                edges = {base_string + 'edge_' + cell_type + '_' + str(start + i): op for
                         op, i in ops}
                config.update(edges)

                if node_idx != 0:
                    # get node idx
                    input_nodes = sorted(list(map(lambda x: x[1], ops)))
                    input_nodes_idx = '_'.join([str(i) for i in input_nodes])
                    config.update({base_string + 'inputs_node_' + cell_type + '_' + str(node_idx + 2): input_nodes_idx})

                start = end
                n += 1
        return config


class BANANASPT(torch.nn.Module):
    def __init__(self, in_features, num_layers, layer_width):
        super(BANANASPT, self).__init__()
        self.lls = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=layer_width, out_features=layer_width) for _ in range(num_layers - 1)]
        )
        # Add the input layer
        self.lls.insert(0, torch.nn.Linear(in_features, layer_width))

        # Add the output layer
        self.lls.append(torch.nn.Linear(layer_width, 1))

    def forward(self, x):
        for ll in self.lls[:-1]:
            x = torch.nn.ReLU()(ll(x))
        return torch.nn.Sigmoid()(self.lls[-1](x)).reshape(-1)
