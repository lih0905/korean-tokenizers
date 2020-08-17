import pandas as pd
columns = ['id','text','label']
train_data = pd.read_csv('data/ratings_train.txt', sep='\t', names=columns, skiprows=1).dropna() # null데이터 삭제
test_data = pd.read_csv('data/ratings_test.txt', sep='\t', names=columns, skiprows=1).dropna()

train_data[['text','label']].to_csv('data/train_data.csv',index=False)
test_data[['text','label']].to_csv('data/test_data.csv',index=False)
