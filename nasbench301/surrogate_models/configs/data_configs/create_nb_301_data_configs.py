import copy
import json

optimizers = ['rs', 'darts', 'random_ws', 'gdas', 'pc_darts', 're', 'tpe', 'de', 'combo', 'bananas', 'drnas']

config_template_all_data = lambda opt: {
    "train_val_test_split_{}".format(opt): {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
        "filepath_regex": "{}/results_*.json".format(opt),
        "type": "all_result_paths"
    }}

config_template_loo = lambda opt: {
    "train_val_test_split_{}".format(opt): {
        "train": 0.9,
        "val": 0.1,
        "test": 0.0,
        "filepath_regex": "{}/results_*.json".format(opt),
        "type": "all_result_paths"
    }}


def create_config(path, optimizers, config_template):
    output = {}
    for opt in optimizers:
        opt_config = config_template(opt)
        output.update(opt_config)
    output['report_freq'] = 100
    json.dump(output, open(path, 'w'))


def create_data_configs():
    # nb301 all optimizer config
    create_config('nb_301.json', optimizers, config_template_all_data)

    # Leave one optimizer out setting
    for opt in optimizers:
        optimizer_list_copy = copy.deepcopy(optimizers)
        optimizer_list_copy.remove(opt)
        create_config('leave_one_optimizer_out/not_{}.json'.format(opt), optimizer_list_copy, config_template_loo)

        create_config('leave_one_optimizer_out/{}.json'.format(opt), [opt], config_template_loo)


if __name__ == "__main__":
    create_data_configs()
