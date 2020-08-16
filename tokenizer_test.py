from konlpy.tag import Mecab, Okt, Komoran, Hannanum, Kkma
from soynlp.word import WordExtractor
from khaiii import KhaiiiApi
from transformers import AutoTokenizer
import argparse

default_text = """손승연은 시대를 잘못타고났다해도

제시는 요즘 힙합트랩 비트와 알앤비가 마구 섞이는 트렌드에 딱 맞는 재목인데
예능에서만 소비되는게 아까움

톤이면 톤, 스킬이면 스킬 외국에서도 보기드문 보컬인데.. 제시제이랑 좀
비슷한거같으면서 더 찐득한

실제로 본인말로는 외국 유명 프로듀서들이 러브콜해서 진출직전이었는데 코로나사태
이후 중지되었다고.."""

parser = argparse.ArgumentParser(description='토크나이저를 테스트할 문장을 입력하세요.')
parser.add_argument('--text', required=False, default=default_text)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
khaiii = KhaiiiApi()
mecab = Mecab()
okt = Okt()
komoran = Komoran()
hannanum = Hannanum()
kkma = Kkma()

text = args.text

print("-"*5,"원본 텍스트", "-"*5)
print(text)

print("-"*5, "Mecab", "-"*5)
print(mecab.morphs(text))

print("-"*5, "Okt", "-"*5)
print(okt.morphs(text))

print("-"*5, "Komoran", "-"*5)
print(komoran.morphs(text))

print("-"*5, "Hannanum", "-"*5)
print(hannanum.morphs(text))

print("-"*5, "Kkma", "-"*5)
print(kkma.morphs(text))

print("-"*5, "Khaiii", "-"*5)
tokens = []
for word in khaiii.analyze(text):
    tokens.extend([str(m).split('/')[0] for m in word.morphs])
print(tokens)

print("-"*5, "bert-base-multilingual-cased", "-"*5)
print(tokenizer.tokenize(text))


