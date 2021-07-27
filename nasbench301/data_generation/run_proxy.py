import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import shutil

from autoPyTorch import AutoNetImageClassificationMultipleDatasets


def evaluate_config(run_id, config_dir, config, logdir):
    torch.backends.cudnn.benchmark = True

    run_id = run_id
    logdir = os.path.join(logdir, "run_" + str(run_id))
    config_dir = config_dir

    autonet_config = {
        "min_workers": 1,
        "budget_type": "epochs",
        "default_dataset_download_dir": "./datasets/",
        "images_root_folders": ["./datasets/cifar10/"],
        "train_metric": "accuracy",
        "additional_metrics": ["cross_entropy"],
        "validation_split": 0.2,
        "use_tensorboard_logger": True,
        "networks": ['darts'],
        "images_shape": [3, 32, 32],
        "log_level": "info",
        "random_seed": 1,
        "run_id": str(run_id),
        "result_logger_dir": logdir,
        "dataloader_worker": 2}

    # Initialize
    autonet = AutoNetImageClassificationMultipleDatasets(**autonet_config)

    if config is None:
        # Read hyperparameter config
        with open(config_dir, "r") as f:
            hyperparameter_config = json.load(f)
    else:
        with open(config, "r") as f:
            hyperparameter_config = json.load(f)
        #hyperparameter_config = json.loads(config)

    # NOTE: 'budget' has to be set here according to the T_max value from the config
    hyperparameter_config["NetworkSelectorDatasetInfo:network"] = "darts"
    print(hyperparameter_config)
    print(autonet.get_hyperparameter_search_space())
    budget = hyperparameter_config['SimpleLearningrateSchedulerSelector:cosine_annealing:T_max']
    print('budget', budget)

    autonet_config = autonet.get_current_autonet_config()
    result = autonet.refit(X_train=np.array([os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "datasets/CIFAR10.csv")]), Y_train=np.array([0]),
                           X_valid=None, Y_valid=None,
                           hyperparameter_config=hyperparameter_config,
                           autonet_config=autonet_config,
                           budget=budget, budget_type="epochs")

    print("Done with refitting.")

    torch.cuda.empty_cache()

    # Score
    with torch.no_grad():
        df = pd.read_csv("./datasets/cifar10/CIFAR_32_test.csv",
                         header=None).values
        X_test = np.array([path for path in df[:, 0]])
        Y_test = np.array(df[:, 1]).astype(np.int32)
        score = autonet.score(X_test=X_test, Y_test=Y_test)
        result["test_accuracy"] = score

    # Dump
    with open(os.path.join(logdir, "final_output.json"), "w+") as f:
        json.dump(result, f)

    return result


def get_config_dir(config_parent_dir, run_id):
    config_dirs = [os.path.join(config_parent_dir, p) for p in os.listdir(config_parent_dir) if p.startswith("config_")]
    config_dirs.sort(key=lambda x: int(x.replace(".json", "").split("_")[-1]))
    return config_dirs[run_id-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit a random config on CIFAR task')
    parser.add_argument("--run_id", type=int, help="An id for the run.")
    parser.add_argument("--config_parent_dir", type=str, help="Path to config.json", default="./configs")
    parser.add_argument("--config", type=str, help="Config as json string", default=None)
    parser.add_argument("--logdir", type=str, help="Directory the results are written to.", default="logs/darts_proxy")
    parser.add_argument("--offset", type=int, help="An id for the run.", default=0)
    args = parser.parse_args()
    run_id = args.run_id + args.offset
    config_dir = get_config_dir(args.config_parent_dir, run_id)
    print(config_dir)

    rundir = "run_"+str(run_id)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if rundir in os.listdir(args.logdir):
        subdirs = os.listdir(os.path.join(args.logdir, rundir))
        if ("final_output.json" in subdirs) and ("bohb_status.json" in subdirs):
            pass
        else:
            shutil.rmtree(os.path.join(args.logdir, rundir))
            evaluate_config(run_id=run_id, config_dir=config_dir, config=args.config, logdir=args.logdir)
    else:
        evaluate_config(run_id=run_id, config_dir=config_dir, config=args.config, logdir=args.logdir)
