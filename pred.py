from torch.utils.data import DataLoader
from model import *
from utils import *


# Parameters
params = {
    'input_dataset':'QSPpred',
    'use_model' : 'QSPpred',
    'n_splits': 5,
    'random_seed': 0,
    'LR': 0.0001,
    'device':'cuda:0',
    'l2_lambda': 0.0001,
    'method_choose': ['tape', 'esm2', 'protTrans'][0],
    'VALID_BATCH': 512,
    'HIDDEN_SIZE': 128,
    'fc_size':128,
    'max_length' : 100,
    'output_file_path':'outputs',
}

datafile_name = f'datasets/{params["input_dataset"]}/pred.csv'
device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
params["device"] = device

sequences, labels, seq_len = prepare_pred_data(datafile_name,params['max_length'])
emb_file = f"{params['output_file_path']}/{params['input_dataset']}/{params['method_choose']}_pred.npy"
pred_data = get_embedding(params["method_choose"], sequences, params['max_length'], params["device"], emb_file)

pred_dataset = SeqDataset(pred_data, seq_len, labels, params["device"])
pred_dataset_loader = DataLoader(dataset=pred_dataset, batch_size=params['VALID_BATCH'], shuffle=False)

emb_hidd = [768, 1280, 1024][[ 'tape', 'esm2', 'protTrans'].index(params['method_choose'])]
model = RNN_module(device, seq_max_len=params['max_length'], in_size=emb_hidd, hidden_size=params['HIDDEN_SIZE'],fc_size=params['fc_size'], dropout=0)
model.to(device)

avg_S = []
for fold in range(params['n_splits']):
    model_name = f'{params["output_file_path"]}/{params["use_model"]}/folds/{params["method_choose"]}_fold{fold+1}.pth'
    model.load_state_dict(torch.load(model_name))
    T, pred, _ = validation(model, device, pred_dataset_loader)
    pred = pred[:, 1]
    avg_S.append(pred)
avg_pred = sum(avg_S) / len(avg_S)
with open(f'{params["output_file_path"]}/{params["input_dataset"]}/pred.txt','w') as f:
        print(avg_pred,file=f)