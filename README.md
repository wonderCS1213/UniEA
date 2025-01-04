# UniEA

[COLING2025] The source codes for `Unifying Dual-Space Embedding for Entity Alignment via Contrastive Learning`.

## Getting Started

### Datasets
We use entity alignment benchmark datasets **OpenEA** which can be downloaded from [OpenEA](https://github.com/nju-websoft/OpenEA). You need to put the prepared data into `../data/` folder.

### Dependencies
+ Python 3
+ PyTorch
+ networkx==2.5.1
+ Scipy
+ Numpy
+ Pandas
+ Scikit-learn

You can automatically download corresponding dependencies by following scripts:
```
conda create -n UniEA python=3.6
conda activate UniEA
conda install -n UniEA pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3.1 -c pytorch # change according to your need here
pip install -r .\requirements.txt
```

### Running
To run UniEA, please use the following scripts (ps: --task is an argument):
```
python train.py --task en_fr_15k
python train.py --task en_de_15k
python train.py --task d_w_15k
python train.py --task d_y_15k
```

To run 5-fold cross-validation, please use the following script:
```
python run_fold.py --task en_fr_15k
```


> If you have any difficulty or question in running code and reproducing experimental results, please email to wangcunda1213@163.com.

## Acknowledgement
We refer to the codes of these repos: GAEA, GCN-Align, OpenEA, MuGNN, IMEA. Thanks for their great contributions!
