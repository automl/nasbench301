# NAS-Bench-301

This repository containts code for the paper: ["NAS-Bench-301 and the Case for Surrogate Benchmarks for Neural Architecture Search"](https://arxiv.org/abs/2008.09777).

The surrogate models can be downloaded on figshare. This includes the models for [v0.9](https://figshare.com/articles/software/nasbench301_models_v0_9_zip/12962432) and [v1.0](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510) as well as the [dataset](https://figshare.com/articles/dataset/NAS-Bench-301_Dataset_v1_0/13246952) that was used to train the surrogate models. We also provide the [full training logs](https://figshare.com/articles/dataset/nasbench301_full_data/13286105) for all architectures, which include learning curves on the train, validation and test sets. These can
be automatically downloaded, please see `nasbench301/example.py`.

To install all requirements (this may take a few minutes), run

```sh
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ pip install nasbench301
```

If installing directly from github
```sh
$ git clone https://github.com/automl/nasbench301
$ cd nasbench301
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ pip install .
```

To run the example
```sh
$ python3 nasbench301/example.py
```

To fit a surrogate model run

```sh
$ python3 fit_model.py --model gnn_gin --nasbench_data PATH_TO_NB_301_DATA_ROOT --data_config_path configs/data_configs/nb_301.json  --log_dir LOG_DIR
```

# Training models from scratch

To create the dataset used for the benchmark we trained the model using a version of AutoPyTorch which can be found here: https://github.com/automl/Auto-PyTorch/tree/nb301.

To train a model with its hyperparameters and architecture described using a configspace representation (`.json` file), firstly download the [CIFAR10 data](https://drive.google.com/file/d/1d5eusa5Pslje99MMEw_VRaYD6oqRNxdU/view?usp=sharing) as used by AutoPyTorch and extract it at `nasbench301/data_generation/datasets`. Then run the following to start the training:

```sh
$ cd nasbench301/data_generation
$ python3 run_proxy.py --run_id 1 --config configs/config_0.json
```

The configuration file can be any other customly generated one. Check the file `nasbench301/representations.py` to convert an object in the `Genotype` representation from DARTS to a `ConfigSpace` object.

## NOTE: This codebase is still subject to changes. Upcoming updates include improved versions of the surrogate models and code for all experiments from the paper. The API may still be subject to changes.
