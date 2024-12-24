import torch
import os
import sys
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.metrics import (matthews_corrcoef, precision_recall_curve, auc, accuracy_score,
                            recall_score, average_precision_score, roc_curve, precision_score, f1_score,roc_auc_score)
import re

class SeqDataset(Dataset):
    def __init__(self, x_data, x_len, x_label, device):
        self.device = device
        self.x_data = torch.tensor(x_data).to(device)
        self.x_len = torch.tensor(x_len).to(device)
        self.x_label = torch.tensor(x_label).to(device)


    def __getitem__(self, index):
        return self.x_data[index].to(self.device), self.x_len[index].to(self.device), self.x_label[index].to(self.device)
    def __len__(self):
        return len(self.x_data)

class SeqDataset_pred(Dataset):
    def __init__(self, x_data, x_len, device):
        self.device = device
        self.x_data = torch.tensor(x_data).to(device)
        self.x_len = torch.tensor(x_len).to(device)
        # print(len(self.x_data), len(self.x_len), len(self.x_label))

    def __getitem__(self, index):
        return self.x_data[index].to(self.device), self.x_len[index].to(self.device)
    def __len__(self):
        return len(self.x_data)

#######################
### Data embeddings ###
#######################

def seq_max_length(seq_list):
    max_length = 0
    for item in seq_list:
            # print(item)
            if len(item) > max_length:
                max_length = len(item)
    return max_length


def seq_to_kmers(seq, k=3):
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


def to_tmp_fasta(sequences,file_name):
    seq_id = 0
    tmp_line = ''
    for seq in sequences:
        seq_id+=1
        tmp_line = tmp_line + '>' + str(seq_id) + '\n' + seq + '\n'
    with open(file_name,'w') as f:
        print(tmp_line.strip(),file=f)


def Onehot_embedding(sequences, max_length):
    count = 0
    Alfabeto = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    sequence_feature = []
    for sequence in sequences:
        count += 1
        data_temp = np.zeros(shape=[max_length, len(Alfabeto)])
        sequence = sequence.upper()
        length_temp = len(sequence)
        indices = [Alfabeto.index(c) for c in sequence if c in Alfabeto]
        data_temp = np.zeros((max_length, len(Alfabeto)))
        for j, index in enumerate(indices):
            data_temp[j, index] = 1.0
        sequence_feature.append(data_temp)
        percent = int((count / len(sequences)) * 100)
        bar = '#' * int(count / len(sequences) * 20)
        print(f'\r[{bar:<20}] {percent:>3}% ({count}/{len(sequences)})', end='')
    sequence_feature = np.array(sequence_feature,dtype=np.float32)
    return sequence_feature



def Tape_embedding(sequences,max_length,device,emb_name):
    sequence_feature = []
    if not os.path.exists(emb_name):
        import biovec
        from Bio.Seq import Seq
        from biovec import models
        from tape import ProteinBertModel, TAPETokenizer
        model = ProteinBertModel.from_pretrained('bert-base')
        tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
        model.eval()
        model.to(device)
        
        count = 0
        for sequence in sequences:
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            tmp_list = [tokenizer.encode(sequence.upper())]
            tmp_array = np.array(tmp_list)
            token_ids = torch.from_numpy(tmp_array).to(device)  # encoder
            sequence_output, _ = model(token_ids)   
            sequence_output = sequence_output.to('cpu').detach().numpy()
            feature = sequence_output.squeeze()
            feature = np.delete(feature,-1,axis=0)
            feature = np.delete(feature,0,axis=0)
            size = feature.shape[0]
            
            pad_length = max_length - size
            if pad_length:
                padding = np.zeros((pad_length,768),dtype='float32')
                feature = np.r_[feature,padding]
            sequence_feature.append(feature)
            count+=1
            percent = int((count / len(sequences)) * 100)
            bar = '#' * int(count / len(sequences) * 20)
            print(f'\r[{bar:<20}] {percent:>3}% ({count}/{len(sequences)})', end='')
        sequence_feature = np.array(sequence_feature,dtype=np.float32)
        if emb_name != 'no':
            np.save(emb_name,sequence_feature)
    else:
        sequence_feature = np.load(emb_name)
    return sequence_feature



def protTrans_emb(sequences,max_length,device, emb_name):
    sequence_feature = []
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    if not os.path.exists(emb_name):
        sequence_feature = []
        with torch.no_grad():
            count = 0
            for seq in sequences:
                sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
                # tokenize sequences and pad up to the longest sequence in the batch
                ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
                input_ids = torch.tensor(ids['input_ids']).to(device)
                attention_mask = torch.tensor(ids['attention_mask']).to(device)
                embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
                # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]) 
                token_representations = embedding_repr.last_hidden_state[0,:len(seq)] # shape (7 x 1024)
                feature = token_representations.to('cpu').detach().numpy()
                pad_length = max_length - len(seq)
                if pad_length:
                    padding = np.zeros((pad_length,1024),dtype='float32')
                    feature = np.r_[feature,padding]
                sequence_feature.append(feature)
                count+=1
                percent = int((count / len(sequences)) * 100)
                bar = '#' * int(count / len(sequences) * 20)
                print(f'\r[{bar:<20}] {percent:>3}% ({count}/{len(sequences)})', end='')
        sequence_feature = np.array(sequence_feature,dtype=np.float32)
        if emb_name != 'no':
            np.save(emb_name,sequence_feature)
    else:
        sequence_feature = np.load(emb_name)
    return sequence_feature


def ESM_embedding(sequences,max_length,device,emb_name):
    sequence_feature = []
    if not os.path.exists(emb_name):
        import esm
        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        model.to(device)
        sequence_feature = []
        count = 0
        for sequence in sequences:
            data = [("seq",sequence)]
            _, _, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            feature = token_representations.to('cpu').detach().numpy()
            pad_length = max_length - len(sequence)
            feature =feature.squeeze(0)
            if pad_length:
                padding = np.zeros((pad_length,1280),dtype='float32')
                feature = np.r_[feature,padding]
            sequence_feature.append(feature)
            count+=1
            percent = int((count / len(sequences)) * 100)
            bar = '#' * int(count / len(sequences) * 20)
            print(f'\r[{bar:<20}] {percent:>3}% ({count+1}/{len(sequences)})', end='')
        sequence_feature = np.array(sequence_feature,dtype=np.float32)
        if emb_name != 'no':
            np.save(emb_name,sequence_feature)
    else:
        sequence_feature = np.load(emb_name)
    return sequence_feature



def get_embedding(emb_method, sequences, max_length, device, emb_file):
    if emb_method == 'tape':
        return Tape_embedding(sequences, max_length, device, emb_file)
    elif emb_method == 'onehot':
        return Onehot_embedding(sequences, max_length)
    elif emb_method == 'esm2':
        return ESM_embedding(sequences, max_length, device, emb_file)
    elif emb_method == 'protTrans':
        return protTrans_emb(sequences, max_length, device, emb_file)
    else:
        raise ValueError(f"Unknown embedding method: {emb_method}")



def train(model,device,train_loader,optimizer,l2_lambda = 0.1):
    model.train()
    loss_sum = []
    loss_function = torch.nn.CrossEntropyLoss()
    for i, (seq, seq_len, labels) in enumerate(train_loader, 0): 
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        seq = seq.to(device)
        labels = labels.long().to(device)
        y_pred = model(seq,seq_len)
        loss = loss_function(y_pred, labels)  + l2_lambda * l2_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum.append(loss.item())
    loss_train = np.average(loss_sum) 
    return loss_train



def validation(model,device,valid_loader,l2_lambda = 0.1):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    loss_function = torch.nn.CrossEntropyLoss()
    loss_sum = []
    for i, (seq, seq_len, labels) in enumerate(valid_loader, 0):  
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        seq = seq.to(device)
        labels = labels.long().to(device)
        seq = seq.to(device)
        y_pred = model(seq,seq_len)
        labels = labels.long().to(device)
        total_preds = torch.cat((total_preds, y_pred.cpu()), 0)
        total_labels = torch.cat((total_labels, labels.cpu()), 0)
        loss = loss_function(y_pred, labels)  + l2_lambda * l2_reg
        loss_sum.append(loss.item())
        
    loss_test = np.average(loss_sum) 
    
    
    total_labels = total_labels.detach().numpy()
    probs = torch.softmax(total_preds,dim=1)
    
    probs_np = probs.detach().numpy()
    return total_labels,probs_np,loss_test




def tsne_validation(model,device,valid_loader,l2_lambda = 0.1):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    loss_function = torch.nn.CrossEntropyLoss()
    loss_sum = []
    total_tsne = torch.Tensor()
    loss_function = torch.nn.BCELoss()
    for i, (seq, seq_len, labels) in enumerate(valid_loader, 0):  
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        seq = seq.to(device)
        labels = labels.long().to(device)
        seq = seq.to(device)
        y_pred,tsne = model(seq,seq_len)
        labels = labels.long().to(device)
        total_preds = torch.cat((total_preds, y_pred.cpu()), 0)
        total_labels = torch.cat((total_labels, labels.cpu()), 0)
        total_tsne = torch.cat((total_tsne, tsne.cpu()), 0)
        loss = loss_function(y_pred, labels)  + l2_lambda * l2_reg
        loss_sum.append(loss.item())
    loss_test = np.average(loss_sum) 
    total_labels = total_labels.detach().numpy()
    probs = torch.softmax(total_preds,dim=1)
    probs_np = probs.detach().numpy()
    return total_labels,probs_np,loss_test,total_tsne





def prepare_pred_data(dataset, max_length=100):
    train_seq_list = open(dataset, 'r', encoding='utf-8').readlines()
    sequences = []
    train_len = []
    labels = [0]*len(train_seq_list)
    for seq in train_seq_list:
        line = seq.strip()
        if len(line)>max_length:
            line = line[:max_length]
        sequences.append(line)
        train_len.append(len(line))
    print('max length:', max_length)
    labels = np.array(labels, dtype=np.float32)
    train_len = np.array(train_len, dtype=np.float32)
    return sequences, labels, train_len


def prepare_data(dataset_path, max_length=100):
    seq_list = open(dataset_path, 'r', encoding='utf-8').readlines()
    sequences = []
    labels = []
    train_len = []
    for seq in seq_list:
        line = seq.strip().split('\t')
        if len(line[0])>max_length:
            line[0] = line[0][:max_length]
        sequences.append(line[0])
        labels.append(int(line[1]))
        train_len.append(len(line[0]))
    print('max length:', max_length)
    labels = np.array(labels, dtype=np.float32)
    train_len = np.array(train_len, dtype=np.float32)
    return sequences, labels, train_len


def run_experiment(model,**kwrags):
    train_data = kwrags["train_data"]
    train_index = kwrags["train_index"]
    device = kwrags["device"]
    labels = kwrags["labels"]
    train_len = kwrags["train_len"]
    valid_index = kwrags["valid_index"]
    dataset_outputs = kwrags["dataset_outputs"]
    train_data = kwrags["train_data"]
    embedding_method = kwrags["method_choose"]
    fold = kwrags["fold"]
    optimizer = torch.optim.Adam(model.parameters(), lr=kwrags['LR'])
    train_dataset = SeqDataset(train_data[train_index], train_len[train_index], labels[train_index], device)
    valid_dataset = SeqDataset(train_data[valid_index], train_len[valid_index], labels[valid_index], device)
    train_loader = DataLoader(dataset=train_dataset, batch_size=kwrags['TRAIN_BATCH'], shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=kwrags['VALID_BATCH'], shuffle=False)
    save_dir = dataset_outputs + f'/folds/'
    os.makedirs(save_dir, exist_ok=True)
    best_auc = 0
    best_accuracy = 0
    best_precison = 0
    best_epoch = None
    epoch_step = 0

    for epoch in range(kwrags['EPOCHES']):
        train_loss = train(model, device, train_loader, optimizer, kwrags['l2_lambda'])
        T, S, val_loss = validation(model, device, valid_loader, kwrags['l2_lambda'])
        S = S[:, 1]
        P = (S > 0.5).astype(int)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, P)
        accuracy = accuracy_score(T, P)
        AUCS = [epoch + 1, train_loss, val_loss, AUC, accuracy, precision]
        print('\t'.join([str(x) for x in AUCS]))

        if accuracy > best_accuracy:
            epoch_step = 0
            best_epoch = epoch + 1
            best_auc = AUC
            best_accuracy = accuracy
            best_precison = precision
            model_file = os.path.join(save_dir, f'{embedding_method}_fold{fold+1}.pth')
            torch.save(model.state_dict(), model_file)
        else:
            epoch_step += 1

        if kwrags['early_stop'] == epoch_step:
            break
    
    return best_epoch, best_auc, best_accuracy, best_precison


def load_test_set(**kwargs):
    sequences, labels, test_len = prepare_data(kwargs['dataset'],kwargs['max_length'])
    emb_file = kwargs['dataset_outputs'] + '/'+ kwargs['method_choose'] + '_test.npy'
    test_data = get_embedding(kwargs['method_choose'], sequences, kwargs['max_length'], kwargs['device'], emb_file)
    test_dataset = SeqDataset(test_data, test_len, labels, kwargs['device'],)
    return DataLoader(dataset=test_dataset, batch_size=kwargs['VALID_BATCH'], shuffle=False)


def evaluate_predictions(T, pred, fold_name=None):
    metrics = {}
    P = (pred > 0.5).astype(int)
    metrics['Accuracy'] = accuracy_score(T, P)
    metrics['Precision'] = precision_score(T, P)
    metrics['Recall'] = recall_score(T, P)
    metrics['F1 score'] = f1_score(T, P)
    fpr, tpr, _ = roc_curve(T, pred)
    prc_precision, prc_recall, _ = precision_recall_curve(T, pred)
    metrics['prc_precision'] = prc_precision
    metrics['prc_recall'] = prc_recall
    metrics['FPR'] = fpr
    metrics['TPR'] = tpr
    metrics['AUC'] = auc(fpr, tpr)
    pre, rec, _ = precision_recall_curve(T, pred)
    metrics['PRC'] = average_precision_score(T, pred)
    metrics['MCC'] = matthews_corrcoef(T, P)
    if fold_name:
        return {fold_name: metrics}
    return metrics
