import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text = [sent_len, batch_size]
        #print_shape('text',text)
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent_len, batch_size, emb_dim]
        #print_shape('embedded', embedded)
        
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        #print_shape('packed_output', packed_output)
        #print_shape('hidden', hidden)
        #print_shape('cell', cell)
        
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #print_shape('output', output)
        #print_shape('output_lengths', output_lengths)        
        
        # output = [sent_len, batch_size, hi_dim * num_directions]
        # output over padding tokens are zero tensors
        # hidden = [num_layers * num_directions, batch_size, hid_dim]
        # cell = [num_layers * num_directions, batch_size, hid_dim]
        
        # concat the final forward and backward hidden layers
        # and apply dropout
        
        #print_shape('hidden[-2,:,:]', hidden[-2,:,:])
        #print_shape('hidden[-1,:,:]', hidden[-1,:,:])
        #cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        #print_shape('cat', cat)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        #print_shape('hidden', hidden)
        # hidden = [batch_size, hid_dim * num_directions]
        
        res = self.fc(hidden)
        #print_shape('res', res)
        return res