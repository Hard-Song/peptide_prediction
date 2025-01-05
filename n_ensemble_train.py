
import os
import torch
from model import *
from utils import *
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt



# default Parameters
params = {
    'datasets':['AIPpred', 'Antiangiopred', 'Anticp', 'AVPpred', 'BBPpred',     'Cellppdmod', 'Hemopi', 'Proteinflam', 'QSPpred'],
    'train':True,
    'n_splits': 5,
    'random_seed': 0,
    'LR': 0.0001,
    'device':'cuda:0',
    'l2_lambda': 0.0001,
    'EPOCHES': 100,
    'early_stop': 10,
    'TRAIN_BATCH': 32,
    'VALID_BATCH': 128,
    'DROP_OUT': 0.4,
    'HIDDEN_SIZE': 128,
    'fc_size':128,
    'max_length' : 100,
    'dataset' : '',
    'output_file_path':'outputs/',
    'emb_methods' : ['tape', 'esm2', 'protTrans'],
    'emb_hiddens' : [768, 1280, 1024],
}

# Feature extraction methods
device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
params["device"] = device
print(device)
skf = StratifiedKFold(n_splits=params['n_splits'], shuffle=True, random_state=params['random_seed'])
if not os.path.exists(params["output_file_path"]):
    os.makedirs(params["output_file_path"])

for emb_method,emb_hidden in zip(params['emb_methods'],params['emb_hiddens']):
    for dataset in params['datasets']:
        dataset_outputs =f"{params['output_file_path']}/{dataset}/" 
        params["dataset_outputs"] = dataset_outputs
        params["dataset"] = dataset
        if params["train"] == True:
            if not os.path.exists(params["dataset_outputs"]):
                os.makedirs(params["dataset_outputs"])
            emb_file = dataset_outputs + '/'+params['method_choose'] + '_train.npy'
            sequences, labels, train_len = prepare_data(f"datasets/{params['dataset']}/train.csv", params['max_length'])
            train_data = get_embedding(params['method_choose'], sequences, params['max_length'], device, emb_file)
            result_file = os.path.join(dataset_outputs, 'results', f'train_{params["method_choose"]}_log.txt')
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            params["labels"] = labels
            params["train_len"] = train_len
            params["train_data"] = train_data
            with open(result_file, 'w') as f:
                f.write(f"Fold\tEpoch\tAUC\tAccucacy\tPrecision\n")
                for fold, (train_index, valid_index) in enumerate(skf.split(train_data, labels)):
                    print(f'Fold {fold + 1}')
                    params["train_index"] = train_index
                    params["valid_index"] = valid_index
                    params["fold"] = fold
                    model = RNN_module(device, seq_max_len=params["max_length"], in_size=emb_hidden, hidden_size=params['HIDDEN_SIZE'], fc_size=params['fc_size'], dropout=params['DROP_OUT'])
                    model = model.to(device)
                    best_epoch, best_auc, best_accuracy, best_precison = run_experiment(model, **params)
                    f.write(f"{fold}\t{best_epoch}\t{best_auc}\t{best_accuracy}\t{best_precison}\n")

        # eval n-fold
        model = RNN_module(device, seq_max_len=params['max_length'], in_size=emb_hidden, hidden_size=params['HIDDEN_SIZE'],fc_size=params['fc_size'], dropout=0)
        model.to(device)
        metrics = {}
        avg_S = []
        T = []
        sequences, labels, test_len = prepare_data(f"datasets/{params['dataset']}/test.csv",params['max_length'])
        emb_file = params['dataset_outputs'] + '/'+ params['method_choose'] + '_test.npy'
        print(len(sequences))
        test_data = get_embedding(params['method_choose'], sequences, params['max_length'], params['device'], emb_file)
        test_dataset = SeqDataset(test_data, test_len, labels, params['device'],)
        test_loader = DataLoader(dataset=test_dataset, batch_size=params['VALID_BATCH'], shuffle=False)
        print(params["dataset_outputs"],params["method_choose"])
        for fold in range(params['n_splits']):
            model.load_state_dict(torch.load(f'{params["dataset_outputs"]}/folds/{params["method_choose"]}_fold{fold+1}.pth'))
            T, pred, _ = validation(model, device, test_loader)
            pred = pred[:, 1]
            avg_S.append(pred)
            fold_metrics = evaluate_predictions(T, pred, fold_name=f'fold{fold+1}')
            metrics.update(fold_metrics)
        avg_pred = sum(avg_S) / len(avg_S)
        ensemble_metrics = evaluate_predictions(T, avg_pred)
        ttt = [avg_pred,T]
        with open(dataset_outputs+'/results/pred.txt','w') as f:
            print(ttt,file=f)
        metrics['fold-ensemble'] = ensemble_metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for label, metric in metrics.items():
            print(label)
            for metric_name, metric_value in metric.items():
                if metric_name in ['FPR','TPR','prc_precision','prc_recall']:
                    continue
                print('%s: %0.2f' % (metric_name, metric_value))
            print('\n')
            fpr = metric['FPR']
            tpr = metric['TPR']
            roc_auc = metric['AUC']
            ax1.plot(fpr, tpr, lw=2, label='%s (AUC = %0.2f)' % (label, roc_auc))
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            prc_precision = metric['prc_precision']
            prc_recall = metric['prc_recall']
            prc_auc = metric['PRC']
            ax2.plot(prc_recall, prc_precision, lw=2, label='%s (PRC = %0.2f)' % (label, prc_auc))
            ax2.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        plt.savefig(f'{dataset_outputs}/results/{params["method_choose"]}_AUPR.png')
        plt.close()
        with open(f'{dataset_outputs}/results/{params["method_choose"]}_test.txt', 'w') as f:
            print(metrics, file=f)
