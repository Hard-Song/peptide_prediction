from torch.utils.data import DataLoader
from model import *
from utils import *
import json

# Parameters
params = {
    'input_dataset':"QSPpred",
    'peptide_models':['AIPpred', 'Antiangiopred', 'Anticp', 'AVPpred', 'BBPpred',     'Cellppdmod', 'Hemopi', 'Proteinflam', 'QSPpred'],
    'max_length' : 100,
    'output_file_path':'outputs',
    'n_splits': 5,
    'random_seed': 0,
    'LR': 0.0001,
    'device':'cuda:0',
    'l2_lambda': 0.0001,
    'emb_methods' : ['tape', 'esm2', 'protTrans'],
    'emb_hiddens' : [768, 1280, 1024],
    'VALID_BATCH': 512,
    'HIDDEN_SIZE': 128,
    'fc_size':128,
}

input_dataset = f'datasets/{params["input_dataset"]}/pred.csv'
device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
params["device"] = device

sequences, labels, seq_len = prepare_pred_data(input_dataset,params['max_length'])
all_pep_pred_results = {}
for emb_method, emb_hidden in zip(params["emb_methods"],params["emb_hiddens"]):
    emb_file = f"{params['output_file_path']}/{params['input_dataset']}/{emb_method}_pred.npy"
    pred_data = get_embedding(emb_method, sequences, params['max_length'], params["device"], emb_file)

    pred_dataset = SeqDataset(pred_data, seq_len, labels, params["device"])
    pred_dataset_loader = DataLoader(dataset=pred_dataset, batch_size=params['VALID_BATCH'], shuffle=False)

    model = RNN_module(device, seq_max_len=params['max_length'], in_size=emb_hidden, hidden_size=params['HIDDEN_SIZE'],fc_size=params['fc_size'], dropout=0)
    model.to(device)
    this_emb_results = {}
    for peptide_model in params['peptide_models']:
        this_pep_pred = []
        for fold in range(params['n_splits']):
            model_name = f'{params["output_file_path"]}/{peptide_model}/folds/{emb_method}_fold{fold+1}.pth'
            model.load_state_dict(torch.load(model_name))
            T, pred, _ = validation(model, device, pred_dataset_loader)
            pred = pred[:, 1]
            this_pep_pred.append(pred)
        this_pep_pred = sum(this_pep_pred) / len(this_pep_pred)
        this_emb_results[peptide_model] = this_pep_pred
    all_pep_pred_results[emb_method] = this_emb_results
# with open(f'{params["output_file_path"]}/{params["input_dataset"]}/pred.json','w') as f:
#     json.dump(all_pep_pred_results,f,ensure_ascii=False,indent=4)

# result to csv
multi_peptide_pred_results = []
for seq in range(len(sequences)):
    head = ["Sequences"]
    multi_peptide_pred = [sequences[seq],]
    for pep_kind in params["peptide_models"]:
        for emb in params["emb_methods"]: 
            head.append([pep_kind,emb])
            multi_peptide_pred.append(all_pep_pred_results[emb][pep_kind][seq])
    multi_peptide_pred_results.append(multi_peptide_pred)
multi_peptide_pred_results.insert(0,head)
with open(f'{params["output_file_path"]}/{params["input_dataset"]}/pred.csv','w') as f:
    for i in multi_peptide_pred_results:
        print(','.join(i),file=f)