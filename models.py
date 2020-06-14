import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_dim, dropout, out_dim):
        """初始化参数：
            vocab_size：字典的大小
            embedding_dim：词向量的维数
            hidden_dim：隐向量的维数
            out_dim：标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=num_layers,
                              dropout=dropout
                              )

        self.line = nn.Linear(2*hidden_dim, out_dim)

    def forward(self, data, lengths):
        embedding = self.embedding(data)  # [B, L, embedding_dim]
        packed = pack_padded_sequence(embedding, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(
            rnn_out, batch_first=True, total_length=178)
        scores = self.line(rnn_out)  # [B, L, out_size]

        return scores
