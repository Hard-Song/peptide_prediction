

# Peptide Classification via multi-embeddings

## Brief introduction

This is a repository regarding the classification of short peptides. Three different protein sequence embedding models are adopted as the initial protein feature representations, and the RNN (Recurrent Neural Network) model is used to further integrate the deep features of proteins, ultimately achieving the purpose of classifying a variety of short peptides.


## Usage

### Requirements
* PyTorch 1.13.1

* transformers

* sklearn

We utlized three embedding methods as follows:

* TAPE(Rao, R. et al [[ paper](https://arxiv.org/abs/1906.08230) | [Gtihub ](https://github.com/li-ziang/tape)]

* ESM-2 (Lin, Z. et al.) [[ paper ](https://www.science.org/doi/abs/10.1126/science.ade2574) | [Github ](https://github.com/facebookresearch/esm)]

* ProtTrans (Elnaggar, A. & Heinzinger, M.,et al.) [[ paper](https://ieeexplore.ieee.org/document/9477085/) | [Github ](https://github.com/agemagician/ProtTrans?tab=readme-ov-file)]

### Dataset structure

Please organize each dataset as the following structure:

```
datasets/
└── dataset/
    ├── train.csv   # protein amino sequence \t label
    ├── test.csv    # protein amino sequence \t label
    └── pred.csv    # protein amino sequence
n_ensemble_train.py
n_ensemble_pred.py
train.py
test.py
pred.py
utils.py
model.py
```

### Train / Evaluate / Predict for single peptide 

Use `python train.py` to start training and set initial parameters in `parms{}` beforehand. The trained models and results are saved in `outputs/dataset/` by default. After training, testing can be done and its results are also saved in the same directory. Additionally, `python pred.py` can be used to predict a single short peptide, leveraging the trained models in `outputs/dataset/` to generate predicted results.


### Train / Evaluate / Predict for multi-peptide 

Use `python n_ensemble_train.py` to start training and set the initial parameters in `parms{}` beforehand. By default, the trained models and results will be saved in `outputs/dataset/`.

After the training of multiple short peptide classifiers is completed, use `python n_ensemble_pred.py` to predict the functions of multiple short peptides for the target sequence. The prediction results will be saved in `outputs/dataset/pred.csv`.

