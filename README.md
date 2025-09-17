#### Auxiliary Behavior‑Driven Causal Modeling for Multi‑Behavior Recommendation

This work is implemented in Python 3

The Pytorch version is available in _requirements.txt_


#### Getting Started

First, run the data_process.py file in data/xxx to collate the data set.

you can run Tmall dataset with:

`python3 main.py --data_name tmall`

or

`./b_tmall.sh`

The optimal hyperparameters for the model are shown in the table below:

| Dataset |  lr  | lambda | reg_weight (beta) |
|:-------:|:----:|:------:|:-----------------:|
|  Tmall  | 1e-4 |  0.5   |       1e-4        |
| Taobao  | 1e-3 |  0.7   |       1e-4        |
|  IJCAI  | 1e-4 |  0.5   |       1e-4        |
# ABDCM
