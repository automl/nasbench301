# NAS-Bench-301

This repository containts code for the paper: ["NAS-Bench-301 and the Case for Surrogate Benchmarks for Neural Architecture Search"](https://arxiv.org/abs/2008.09777).

The surrogate models can be downloaded on figshare. This includes the models for [v0.9](https://figshare.com/articles/software/nasbench301_models_v0_9_zip/12962432) and [v1.0](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510).

To install all requirements (this may take a few minutes), run

```sh
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ pip install nasbench301
```

To run a quick example, adapt the model paths in 'nasbench301/example.py' and from the base directory run

```sh
$ export PYTHONPATH=$PWD
$ python3 nasbench301/example.py
```

## NOTE: This codebase is still subject to changes. Upcoming updates include improved versions of the surrogate models and code for all experiments from the paper. The API may still be subject to changes.
