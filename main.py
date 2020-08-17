import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from konlpy.tag import Mecab, Okt, Komoran, Hannanum, Kkma
from khaiii import KhaiiiApi

from data_loader import dataloader
from model import RNN
from utils import train, evaluate, binary_accuracy, epoch_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="하이퍼 파라미터 설정")

# tokenizers : 'space', 'character', 'khaiii', 'mecab', 'okt', 'komoran', 'kkma'
khaiii = KhaiiiApi()
def khaiii_tokenize(text):
    tokens = []
    for word in khaiii.analyze(text):
        tokens.extend([str(m).split('/')[0] for m in word.morphs])
    return tokens

mecab = Mecab().morphs
okt = Okt().morphs
komoran = Komoran().morphs
hannanum = Hannanum().morphs # 오류 발생 
kkma = Kkma().morphs

def space_tokenizer(text):
    return text.split(' ')

def char_tokenizer(text):
    return [t for t in text]

tokenizers = [space_tokenizer, char_tokenizer, khaiii_tokenize, mecab, okt, komoran, kkma]
tokenizer_names = ['space', 'character', 'khaiii', 'mecab', 'okt', 'komoran', 'kkma']


if __name__ == '__main__':
    
    # hyperparameters
    parser.add_argument('--n_epochs', required=False, default=10, type=int)
    parser.add_argument('--max_vocab_size', required=False, default=30000, type=int)
    parser.add_argument('--batch_size', required=False, default=64, type=int)
    parser.add_argument('--embedding_dim', required=False, default=100, type=int)
    parser.add_argument('--hidden_dim', required=False, default=256, type=int)
    parser.add_argument('--n_layers', required=False, default=2, type=int)
    parser.add_argument('--bidirectional', required=False, default=True, type=bool)
    parser.add_argument('--dropout', required=False, default=0.5, type=float)

    args = parser.parse_args()
    print(args)
    
    # save the result into a dictionary
    result = dict()

    # training and evaluate
    for tokenizer_name, tokenizer in zip(tokenizer_names, tokenizers):
        print(f'-------------------------------------------------------------')
        print(f'Data loading with {tokenizer_name} tokenizer...')
        start_time = time.time()
        TEXT, LABEL, train_iterator, test_iterator = dataloader(tokenizer,
                                                                args.max_vocab_size,
                                                                args.batch_size, device)
        input_dim = len(TEXT.vocab)
        print(f'The number of vocabularies is {input_dim}.')

        end_time = time.time()
        data_loading_time = round(end_time - start_time,3)
        data_prep_mins, data_prep_secs = epoch_time(start_time, end_time)
        print(f'Data loading Time: {data_prep_mins}m {data_prep_secs}s')


        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        model = RNN(input_dim, args.embedding_dim,
                    args.hidden_dim, 1, args.n_layers, 
                    args.bidirectional, args.dropout, pad_idx)
        model.embedding.weight.data[pad_idx] = torch.zeros(args.embedding_dim)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        best_test_loss = float('inf')

        loss_result = []
        acc_result = []
        elapsed_time = []

        print(f'Training with {tokenizer_name} tokenizer...')
        for epoch in range(args.n_epochs):

            start_time = time.time()

            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
            loss_result.append(round(test_loss,3))
            acc_result.append(round(test_acc,3))

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            elapsed_time.append(round(end_time - start_time,3))

            if test_loss < best_test_loss:
                best_test_loss = test_loss

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

        result[tokenizer_name] = {'loss':loss_result, 'acc':acc_result, 'training time':elapsed_time, 'data loading time':data_loading_time, 'num. vocabs':input_dim}

    print('The final result is...')
    print(result)

    # save the result into 'result.json'
    with open('result.json', 'w') as f:
        json.dump(result, f)