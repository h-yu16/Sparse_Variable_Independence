# Sparse Variable Independece

Official repo for AAAI 2023 paper [Stable Learning via Sparse Variable Independence](https://arxiv.org/abs/2212.00992).

## Quick Start

### Running SVI

```bash
CUDA_VISIBLE_DEVICES=0 python3 exp_svi.py 
```

### Running other algorithms

#### OLS

```bash
CUDA_VISIBLE_DEVICES=0 python3 exp_linear.py --reweighting None --paradigm regr
```

#### STG

```bash
CUDA_VISIBLE_DEVICES=0 python3 exp_linear.py --reweighting None --paradigm fs
```

#### DWR

```bash
CUDA_VISIBLE_DEVICES=0 python3 exp_linear.py --reweighting DWR --paradigm regr
```

#### SVI $^d$

```bash
CUDA_VISIBLE_DEVICES=0 python3 exp_linear.py --reweighting DWR --paradigm fs
```

## Citing

If you find this repo useful for your research, please consider citing the paper.

```
@article{yu2022stable,
  title={Stable Learning via Sparse Variable Independence},
  author={Yu, Han and Cui, Peng and He, Yue and Shen, Zheyan and Lin, Yong and Xu, Renzhe and Zhang, Xingxuan},
  journal={arXiv preprint arXiv:2212.00992},
  year={2022}
}
```
