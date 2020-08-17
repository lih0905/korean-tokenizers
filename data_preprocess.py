import pandas as pd
columns = ['id','text','label']
train_data = pd.read_csv('data/ratings_train.txt', sep='\t', names=columns, skiprows=1).dropna() # null데이터 삭제
test_data = pd.read_csv('data/ratings_test.txt', sep='\t', names=columns, skiprows=1).dropna()

# Hannanum 및 Kkma 토크나이저 에러나는 라인 삭제
err_ind1 = train_data[train_data['text']=='ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ'].index[0]
err_ind2 = train_data[train_data['text']=='ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ'].index[0]
err_ind3 = test_data[test_data['text']=='진짜 조낸 재밌다 굿굿굿굿굿굿굿굿굿굿굿굿굿굿굿굿굿굿굿굿굿굿'].index[0]
err_ind4 = test_data[test_data['text']=='황시욬황시욬황시욬황시욬황시욬황시욬황시욬황시욬황시욬황시욬'].index[0]
err_ind5 = test_data[test_data['text']=='ㄹㅇㄴㅁㄹㅇㄴㅁㄹㅇㄴㅁㄹㅇㄴㅁㄹㅇㄴㅁㄹㅇㄴㅁㄹㅇㄴㅁㄹㅇㄴㅁㄹㅇㄴㅁ'].index[0]

train_data = train_data.drop([err_ind1, err_ind2]) 
test_data = test_data.drop([err_ind3, err_ind4, err_ind5])

# CSV 파일로 저장
train_data[['text','label']].to_csv('data/train_data.csv',index=False)
test_data[['text','label']].to_csv('data/test_data.csv',index=False)
