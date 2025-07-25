# -*- coding: utf-8 -*-
"""transformer_임시.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Cy32yLSDIk3SVSW6s6wOg9oi-L4wZnCQ
"""


# 최종 통합 코드 (Colab friendly: 학습 + 시각화)

#참고코드 https://github.com/datnnt1997/multi-head_self-attention/blob/master/SelfAttention.ipynb
#이론 참조 깃허브 https://github.com/IAAR-Shanghai/Awesome-Attention-Heads


import torch                      # PyTorch 핵심 모듈
import torch.nn as nn             # 신경망 구성용 모듈
import torch.nn.functional as F  # 활성함수 등
import pandas as pd              # 표 형태 출력을 위한 모듈
from transformers import GPT2Tokenizer  # 토큰화를 위한 HuggingFace GPT2 tokenizer
from torch.utils.data import DataLoader, Dataset  # 데이터셋 관리
from sklearn.metrics.pairwise import cosine_similarity
import random

# GPU 설정 (가능하면 CUDA 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 어텐션 계산 함수: scaled dot-product attention
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5  # 유사도 점수 계산 후 scaling
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)  # attention 가중치
    output = torch.matmul(attn, v)    # 가중합 결과 출력
    return output, attn

# 멀티헤드 셀프 어텐션 클래스 정의
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0  # head 수가 나누어 떨어져야 함
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V를 위한 선형변환
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)  # 출력 결합

    def forward(self, x):
        B, T, C = x.size()  # 배치크기, 시퀀스길이, 임베딩 차원
        # Q, K, V 생성 후 head 차원 추가
        Q = self.q_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # 어텐션 수행
        out, attn = scaled_dot_product(Q, K, V)
        # 다시 원래 차원으로 복원
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.fc_out(out), attn

# Transformer 스타일의 작은 언어모델
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)  # 단어 임베딩
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)  # 셀프 어텐션
        #self.norm = nn.LayerNorm(embed_dim)   #정규화
        self.lm_head = nn.Linear(embed_dim, vocab_size)  # 출력 예측 (language modeling head)

    def forward(self, x):
        x = self.token_emb(x)  # [B, T, C]
        x, attn = self.attn(x)  # 정규화 X는 이줄 활성화 후 밑 3줄, 위에 정규화 파트 주석처리
        #x_norm = self.norm(x)               # 정규화
        #x_attn, attn = self.attn(x_norm)    # 어텐션 결과와 가중치
        #x = x + x_attn                      #  Residual 연결
        logits = self.lm_head(x)  # 다음 토큰 예측
   

        return logits, attn

from datasets import load_dataset

# TinyStories 데이터셋 직접 불러오기 (샘플 50000개만 사용)
dataset_full = load_dataset("roneneldan/TinyStories", split="train")
# 2회자, 재다운로드 방지
# 앞에서 5000개만 뽑기
dataset = dataset_full.select(range(50000))
# 텍스트 추출
text = "\n".join(example["text"] for example in dataset)




# GPT2 토크나이저 사용
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # pad token 설정 (eos로 대체)
# 텍스트를 token ID로 인코딩
encodings = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
input_ids = encodings["input_ids"]  # [num_sentences, seq_len]

'''
import re
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

# 간단한 pre-tokenization: 공백 + 특수문자 기준으로 토큰 분리
def simple_pre_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# merge 함수: (a,b) 쌍을 new_index로 묶기
def merge(indices, pair, new_index):
    merged = []
    i = 0
    while i < len(indices):
        if i < len(indices) - 1 and (indices[i], indices[i+1]) == pair:
            merged.append(new_index)
            i += 2
        else:
            merged.append(indices[i])
            i += 1
    return merged

# BPE 학습 함수
def train_bpe(text: str, num_merges: int):
    indices = list(text.encode("utf-8"))
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}

    for i in range(num_merges):
        counts = defaultdict(int)
        for a, b in zip(indices, indices[1:]):
            counts[(a, b)] += 1
        if not counts:
            break
        pair = max(counts, key=counts.get)
        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        indices = merge(indices, pair, new_index)

    return {"vocab": vocab, "merges": merges}

# 개선된 BPE 토크나이저
class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab            # {int: bytes}
        self.merges = merges          # {(a,b): new_index}
        self.inverse_vocab = {v: k for k, v in vocab.items()}  # bytes -> int

    def encode(self, text: str) -> list[int]:
        tokens = simple_pre_tokenize(text)  # 공백, 구두점 등 기준으로 토큰화
        ids = []
        for token in tokens:
            byte_ids = list(token.encode("utf-8"))
            for (a, b), new_id in self.merges.items():
                byte_ids = merge(byte_ids, (a, b), new_id)
            ids.extend(byte_ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors="ignore")
'''


# PyTorch Dataset 정의
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]  # 입력: 앞 n-1개
        y = self.data[idx, 1:]   # 정답: 다음 토큰
        return x, y

# DataLoader 생성
dataset = SimpleDataset(input_ids)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 모델 초기화 및 옵티마이저 설정
vocab_size = tokenizer.vocab_size
model = TinyTransformer(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# 간단한 학습 루프 (3 epoch)
model.train()
for epoch in range(10):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)  # 예측
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))  # next token loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss = {total_loss / len(dataloader):.4f}")

# 데모 입력 문장으로 attention 확인
model.eval()
demo_text1 = "Israel England Syria Iran" #이곳에 단어 입력 ---------------------------------------------->
demo_tokens = tokenizer.tokenize(demo_text1)  # 토큰 문자열
# 토크나이징 및 텐서화
demo_inputs = tokenizer(demo_text1, return_tensors="pt").to(device)
with torch.no_grad():
    _, demo_attn = model(demo_inputs["input_ids"])  # 어텐션 가중치 추출

# attention 가중치 표 출력 함수 정의
def print_attention_avg(attn_weights, tokens, batch=0):
    attn = attn_weights[batch].mean(dim=0).detach().cpu().numpy()
    df = pd.DataFrame(attn, index=tokens, columns=tokens)
    print(f"\n🔹 Attention Heads Average Weights:")
    print(df.round(2))

def print_attention_all_heads(attn_weights, tokens, batch=0):
    num_heads = attn_weights.shape[1]
    for h in range(num_heads):
        attn = attn_weights[batch, h].detach().cpu().numpy()  # Head h에 대한 attention matrix
        df = pd.DataFrame(attn, index=tokens, columns=tokens)
        print(f"\n🔹 Head {h} Attention Weights:")
        print(df.round(2))

def print_attention_focus_avg(attn_weights, tokens, batch=0):
    """
    어텐션 가중치에서 batch=0의 평균 헤드 기준으로
    첫 번째 토큰이 나머지 3개 토큰(1~3)에 얼마나 집중하는지를 백분율로 출력
    """
    attn = attn_weights[batch].mean(dim=0).detach().cpu().numpy()  # Head 평균
    values = attn[0, 1:]  # 첫 번째 토큰이 1~3번째에 주는 가중치
    total = values.sum()
    percentages = (values / total) * 100

    df = pd.DataFrame([percentages], columns=tokens[1:], index=[f"{tokens[0]} →"])
    print(f"\n🔹 Head 평균 기준 '{tokens[0]}' → 나머지 3개 토큰 집중도 (%):")
    print(df.round(1))

def print_token_cosine_similarity(text, model, tokenizer):
    """
    주어진 텍스트 내 각 토큰의 **문맥 반영 임베딩 벡터** 간 코사인 유사도를 계산하고 표로 출력한다.
    """
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        x = model.token_emb(input_ids)  # [1, T, C]
        contextual_embeddings, _ = model.attn(x)  # [1, T, C]
        embeddings = contextual_embeddings[0]     # 시퀀스 차원만 추출 (T, C)

    embed_np = embeddings.cpu().numpy()
    sim_matrix = cosine_similarity(embed_np)

    df = pd.DataFrame(sim_matrix, index=tokens, columns=tokens)
    print(f"\n🔹 Cosine Similarity Between Tokens (Contextualized):")
    print(df.round(2))

def print_cosine_focus_on_first_token(text, model, tokenizer):
    """
    주어진 텍스트에서 첫 번째 토큰이 나머지 토큰들과 얼마나 유사한지를
    문맥 반영 임베딩을 기준으로 코사인 유사도 백분율로 출력.
    """
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        x = model.token_emb(input_ids)  # [1, T, C]
        contextual_embeddings, _ = model.attn(x)  # [1, T, C]
        embeddings = contextual_embeddings[0]     # [T, C]

    embed_np = embeddings.cpu().numpy()
    sim_matrix = cosine_similarity(embed_np)

    first_token_sims = sim_matrix[0, 1:]  # 첫 번째 토큰이 나머지 토큰들과 얼마나 유사한지
    total = first_token_sims.sum()

    if total == 0:
        percentages = [0] * len(first_token_sims)
    else:
        percentages = (first_token_sims / total) * 100

    df = pd.DataFrame([percentages], columns=tokens[1:], index=[f"{tokens[0]} →"])
    print(f"\n🔹 문맥 임베딩 기준 '{tokens[0]}' → 나머지 토큰 유사도 집중도 (%):")
    print(df.round(1))

# 데모 입력에 대한 Head 평균의 어텐션 표 출력
print_attention_avg(demo_attn, demo_tokens, batch=0)

print_attention_focus_avg(demo_attn, demo_tokens)
print_token_cosine_similarity(demo_text1, model, tokenizer)
print_cosine_focus_on_first_token(demo_text1, model, tokenizer)

demo_text2 = "Israel England France Iran" #이곳에 단어 입력 --------------------------------------------->
demo_tokens = tokenizer.tokenize(demo_text2)  # 토큰 문자열
# 토크나이징 및 텐서화
demo_inputs = tokenizer(demo_text2, return_tensors="pt").to(device)
with torch.no_grad():
    _, demo_attn = model(demo_inputs["input_ids"])  # 어텐션 가중치 추출

# 데모 입력에 대한 Head 평균의 어텐션 표 출력
print_attention_avg(demo_attn, demo_tokens, batch=0)

print_attention_focus_avg(demo_attn, demo_tokens)
print_token_cosine_similarity(demo_text2, model, tokenizer)
print_cosine_focus_on_first_token(demo_text2, model, tokenizer)
