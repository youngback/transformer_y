# run_tiny_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scaled Dot-Product Attention
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn

# Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        Q = self.q_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out, attn = scaled_dot_product(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.fc_out(out), attn

# Transformer 모델 정의
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)
        x, attn = self.attn(x)
        logits = self.lm_head(x)
        return logits, attn

# 데이터셋
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y

def main():
    print("[+] Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(50000))  # 먼저 자르고
    texts = [example["text"] for example in dataset]

    print("[+] Tokenizing...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    all_input_ids = []
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        all_input_ids.append(enc["input_ids"])
    input_ids = torch.cat(all_input_ids, dim=0)

    # Dataset 및 Dataloader 정의
    dataset = SimpleDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print("[+] Building model...")
    model = TinyTransformer(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    print("[+] Training...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {total_loss / len(dataloader):.4f}")

    # 테스트 문장
    demo_text = "I think the word that most related to 'Israel' in the 'England', 'Syria', and 'Iran' is a '"
    demo_inputs = tokenizer(demo_text, return_tensors="pt").to(device)
    input_ids = demo_inputs["input_ids"]
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids)
    probs = F.softmax(logits[0, -1], dim=-1)
    next_token_id = torch.argmax(probs).item()
    next_token = tokenizer.decode([next_token_id])
    print(f"\n[next word prediction]: {next_token}")

if __name__ == "__main__":
    main()