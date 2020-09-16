# NAS-Bench-301

This repository containts the code for the paper: ["NAS-Bench-301 and the Case for Surrogate Benchmarks for Neural Architecture Search"](https://arxiv.org/abs/2008.09777).

The surrogate models for the arXiv version v0.9 can be downloaded via ...

To install all requirements (this may take a few minutes), run

```sh
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ pip install torch-scatter==2.0.4+cu102 torch-sparse==0.6.3+cu102 torch-cluster==1.5.5+cu102 torch-spline-conv==1.2.0+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
```

To run a quick example, adapt the model paths in 'nasbench301/example.py' and from the base directory run

```sh
$ export PYTHONPATH=$PWD
$ python3 nasbench301/example.py
```

## NOTE: This codebase is still subject to changes. Upcoming updates include improved version of the surrogate models and code for all experiments from the paper. The API may still be subject to changes.
