# 한국어 토크나이저 비교

[네이버 영화리뷰 데이터](https://github.com/e9t/nsmc)의 분류 모델이 한국어 토크나이저에 따라 어떻게 성능이 달라지는지 살펴본다.

* 모델 : 2-layer 2-directional LSTM with dropout and linear layer

* 토크나이저 
    * 공백 기준
    * 음절 기준
	* Khaiii
    * Kkma
    * Komoran 
    * Mecab 
    * Okt 
    
    * Hannanum -> 오류가 발생하여 생략

* Requirements

```
Python==3.7.3
khaiii==0.4
konlpy==0.5.2
numpy==1.18.5
pandas==1.0.4
torch==1.5.0
torchtext==0.6.0
tqdm==4.46.1
transformers==3.0.2
```    

* Usage

네이버 영화 리뷰 데이터를 다운 받아 data 폴더 안에 저장한 후 다음 코드를 차례대로 실행한다.

```shell
python data_preprocess.py
python main.py [--n_epochs N_EPOCHS] [--max_vocab_size MAX_VOCAB_SIZE]
               [--batch_size BATCH_SIZE] [--embedding_dim EMBEDDING_DIM]
               [--hidden_dim HIDDEN_DIM] [--n_layers N_LAYERS]
               [--bidirectional BIDIRECTIONAL] [--dropout DROPOUT]
```

모든 토크나이저에 대해 훈련이 끝나고 나면 훈련 결과가 `result.json` 파일로 저장된다.

* To Do
    * 자모 단위, BPE 추가
    * Result 추가