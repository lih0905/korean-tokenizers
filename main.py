import time

import torch
import torch.nn as nn
import torch.optim as optim
from konlpy.tag import Mecab, Okt, Komoran, Hannanum, Kkma
from khaiii import KhaiiiApi

from data_loader import dataloader
from model import RNN
from utils import train, evaluate, binary_accuracy, epoch_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
n_epochs = 5
max_vocab_size = 25000
batch_size = 64
embedding_dim = 100
hidden_dim = 256
n_layers = 2
bidirectional = True
dropout = 0.5


# tokenizers
khaiii = KhaiiiApi()
def khaiii_tokenize(text):
    tokens = []
    for word in khaiii.analyze(text):
        tokens.extend([str(m).split('/')[0] for m in word.morphs])
    return tokens

mecab = Mecab()
mecab = mecab.morphs

okt = Okt()
okt = okt.morphs

komoran = Komoran()
komoran = komoran.morphs

hannanum = Hannanum()
hannanum = hannanum.morphs

kkma = Kkma()
kkma = kkma.morphs

tokenizers = [khaiii_tokenize, mecab, okt, komoran, hannanum, kkma]
tokenizer_names = ['khaiii', 'mecab', 'okt', 'komoran', 'hannanum', 'kkma']

for tokenizer_name, tokenizer in zip(tokenizer_names, tokenizers):
    print(f'Data loading with {tokenizer_name} tokenizer...')
    TEXT, LABEL, train_iterator, test_iterator = dataloader(tokenizer,
                                                            max_vocab_size,
                                                            batch_size, device)
    input_dim = len(TEXT.vocab)
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    model = RNN(input_dim, embedding_dim,
                hidden_dim, 1, n_layers, 
                bidirectional, dropout, pad_idx)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')

    print(f'Training with {tokenizer_name} tokenizer...')
    for epoch in range(n_epochs):
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_iterator, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Test Loss: {valid_loss:.3f} |  Test Acc: {valid_acc*100:.2f}%')
