from collections import namedtuple

from ConfigSpace.read_and_write import json as cs_json

import nasbench301 as nb

# Load the performance surrogate model
#NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
print("==> Loading performance surrogate model...")
ensemble_dir_performance = "/path/to/nasbench301_models_v0.9/xgb_v0.9"
performance_model = nb.load_ensemble(ensemble_dir_performance)

# Load the runtime surrogate model
print("==> Loading runtime surrogate model...")
ensemble_dir_runtime = "/path/to/nasbench301_models_v0.9/lgb_runtime_v0.9"
runtime_model = nb.load_ensemble(ensemble_dir_runtime)

# Option 1: Create a DARTS genotype
print("==> Creating test configs...")
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
genotype_config = Genotype(
        normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
        normal_concat=[2, 3, 4, 5],
        reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
        reduce_concat=[2, 3, 4, 5]
        )

# Option 2: Sample from a ConfigSpace
with open("configspace.json", "r") as f:
    json_string = f.read()
    configspace = cs_json.read(json_string)
configspace_config = configspace.sample_configuration()

# Predict
print("==> Predict runtime and performance...")
prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=True)
prediction_configspace = performance_model.predict(config=configspace_config, representation="configspace", with_noise=True)

runtime_genotype = runtime_model.predict(config=genotype_config, representation="genotype")
runtime_configspace = runtime_model.predict(config=configspace_config, representation="configspace")

print("Genotype architecture performance: %f, runtime %f" %(prediction_genotype, runtime_genotype))
print("Configspace architecture performance: %f, runtime %f" %(prediction_configspace, runtime_configspace))
