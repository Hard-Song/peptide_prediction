import torch
import torch.nn as nn

# Models


class RNN_module(torch.nn.Module):
    def __init__(self,device,output=2,seq_max_len=64,in_size=768,hidden_size=128,fc_size=128,dropout=None):
        super(RNN_module, self).__init__()
        self.device = device
        self.max_length = seq_max_len
        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(in_size,hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, fc_size)
        self.fc2 = torch.nn.Linear(fc_size, output)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.num_layers = 1
    def forward(self,seq,seq_len):
        seq_lengths, pro_idx_sort = torch.sort(seq_len,descending=True)[::-1][1], torch.argsort(
            -seq_len)
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        seq = seq.index_select(0, pro_idx_sort)
        x = torch.nn.utils.rnn.pack_padded_sequence(seq, seq_lengths.cpu(), batch_first=True)
        x,_ = self.rnn(x)
        x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.max_length)[0]
        x = x.index_select(0, pro_idx_unsort)
        x = x.mean(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# class nRNN_module(torch.nn.Module):
#     def __init__(self, device, output=2, seq_max_len=64, in_size=768, hidden_size=128, fc_size=128, num_layers=1, dropout=0.2, bidirectional=False):
#         super(nRNN_module, self).__init__()
#         self.device = device
#         self.max_length = seq_max_len
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = torch.nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
#         self.fc1 = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), fc_size)
#         self.fc2 = torch.nn.Linear(fc_size, output)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.relu = nn.LeakyReLU()

#     def forward(self, seq, seq_len):
#         seq_lengths, pro_idx_sort = seq_len.sort(descending=True)
#         pro_idx_unsort = pro_idx_sort.argsort()
#         seq = seq.index_select(0, pro_idx_sort)
#         x = torch.nn.utils.rnn.pack_padded_sequence(seq, seq_lengths.cpu(), batch_first=True)
#         x, _ = self.rnn(x)
#         x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.max_length)
#         x = x.index_select(0, pro_idx_unsort)
#         x = x.mean(1)
#         x = self.fc1(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x


