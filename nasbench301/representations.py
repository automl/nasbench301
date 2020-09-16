from collections import namedtuple


OPS = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
       ]

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def convert_genotype_to_config(arch):
    """Converts a DARTS genotype to a configspace instance dictionary"""
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
                config.update({base_string + 'inputs_node_' + cell_type + '_' + str(node_idx + 2):
                                   input_nodes_idx})

            start = end
            n += 1
    return config


class BaseConverter():
    """Base class for converters from one representation to a dictionary"""

    def __init__(self, name):
        self.name = name

    def convert(self, config):
        raise NotImplementedError("Child classes have to implement convert.")


class ConfigspaceInstanceConverter(BaseConverter):
    """Converter for a ConfigSpace sample to dictionary. Does nothing if it receives a dict"""
    
    def __init__(self, name="configspace"):
        super().__init__(name)

    def convert(self, config):
        if isinstance(config, dict):
            return config
        return config.get_dictionary()


class GenotypeConverter(BaseConverter):
    """Converter for the DARTS genotype."""
    
    def __init__(self, name="genotype"):
        super().__init__(name)

    def convert(self, config):
        config_dict = convert_genotype_to_config(config)
        return config_dict


class BANANASConverter(BaseConverter):
    """BANANAS representation is like the DARTS genotype, but the tuples are inverted (first node, then operation)."""

    def __init__(self, name="BANANAS"):
        super().__init__(name)

    def convert(self, config):
        # Convert to genotype
        normal = [(OPS[op], node) for node, op in bananas_arch[0]]
        reduction = [(OPS[op], node) for node, op in bananas_arch[1]]
        concat = list(range(2, 6))
        genotype = Genotype(normal=normal, reduce=reduction, normal_concat=concat, reduce_concat=concat)
        
        # Convert genotype to configspace dictionary
        config_dict = convert_genotype_to_config(genotype)
        return config_dict


CONVERTER_DICT = {
        "configspace" : ConfigspaceInstanceConverter,
        "genotype" : GenotypeConverter,
        "BANANAS" : BANANASConverter
        }
