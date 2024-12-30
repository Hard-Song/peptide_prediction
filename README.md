# Peptide Classification via multi-embeddings

## Brief introduction

This is a repository regarding the classification of short peptides. Three different protein sequence embedding models are adopted as the initial protein feature representations, and the RNN (Recurrent Neural Network) model is used to further integrate the deep features of proteins, ultimately achieving the purpose of classifying a variety of short peptides.

## Data Source
The relevant data were obtained from the papers and databases corresponding to the functional peptides. For more details, please see the Table S1 attached to the work.

## Usage

### Requirements
* PyTorch 1.13.1

* transformers

* sklearn

We utlized three embedding methods as follows:

* TAPE([Rao, R. et al](https://arxiv.org/abs/1906.08230))  https://github.com/li-ziang/tape

* ESM-2([Lin, Z. et al.](https://www.science.org/doi/abs/10.1126/science.ade2574))   https://github.com/facebookresearch/esm

* ProtTrans([Elnaggar, A. & Heinzinger, M.,et al.](https://ieeexplore.ieee.org/document/9477085/))   https://github.com/agemagician/ProtTrans?tab=readme-ov-file

### Dataset structure

Please organize each dataset as the following structure:

```
datasets/
└── dataset/
    ├── train.csv   # protein amino sequence \t label
    ├── test.csv    # protein amino sequence \t label
    └── pred.csv    # protein amino sequence
train.py
test.py
pred.py
utils.py
model.py
```

### Training & evaluation

Use `python train.py` to start the training. Before training, you can set the initial parameters in `parms{}`. The trained models and results will be saved in `outputs/dataset/` by default.

### Predicting

Use `python pred.py` to start the predicting. Before predicting, you can set the initial parameters in `parms{}`. The trained models and results will be saved in `outputs/dataset/` by default.


