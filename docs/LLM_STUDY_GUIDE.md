# LLM (Large Language Model) 학습 가이드

이 문서는 본 프로젝트의 한국어 GPT 구현을 기반으로 LLM의 핵심 개념을 설명합니다.

---

## 목차

1. [Transformer 아키텍처 개요](#1-transformer-아키텍처-개요)
2. [GPT: Decoder-only 구조](#2-gpt-decoder-only-구조)
3. [임베딩 (Embedding)](#3-임베딩-embedding)
4. [Self-Attention 메커니즘](#4-self-attention-메커니즘)
5. [Feed-Forward Network](#5-feed-forward-network)
6. [Transformer Block](#6-transformer-block)
7. [학습 (Training)](#7-학습-training)
8. [텍스트 생성 (Inference)](#8-텍스트-생성-inference)
9. [토크나이저 (Tokenizer)](#9-토크나이저-tokenizer)
10. [주요 하이퍼파라미터](#10-주요-하이퍼파라미터)
11. [추가 학습 자료](#11-추가-학습-자료)

---

## 1. Transformer 아키텍처 개요

### 1.1 Transformer란?

Transformer는 2017년 Google의 "Attention Is All You Need" 논문에서 제안된 아키텍처입니다.
기존 RNN/LSTM의 순차적 처리 한계를 극복하고, **Self-Attention** 메커니즘을 통해 병렬 처리가 가능합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer 구조                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐              ┌─────────────┐              │
│   │   Encoder   │              │   Decoder   │              │
│   │             │──Attention──▶│             │              │
│   │  (BERT 등)  │              │  (GPT 등)   │              │
│   └─────────────┘              └─────────────┘              │
│                                                             │
│   - 양방향 문맥 이해            - 단방향 (왼→오) 생성        │
│   - 분류, NER 등에 적합         - 텍스트 생성에 적합         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 주요 모델 분류

| 유형 | 구조 | 대표 모델 | 주요 용도 |
|------|------|-----------|-----------|
| Encoder-only | 양방향 | BERT, RoBERTa | 분류, NER, QA |
| Decoder-only | 단방향 (Causal) | GPT, LLaMA | 텍스트 생성 |
| Encoder-Decoder | 양방향 + 단방향 | T5, BART | 번역, 요약 |

---

## 2. GPT: Decoder-only 구조

### 2.1 GPT의 특징

GPT(Generative Pre-trained Transformer)는 **Decoder-only** 구조로,
이전 토큰들만 보고 다음 토큰을 예측하는 **자기회귀(Autoregressive)** 방식입니다.

```
입력:  "오늘 날씨가"
       ↓
GPT:   P(다음 토큰 | "오늘 날씨가")
       ↓
출력:  "좋다" (가장 확률 높은 토큰)
```

### 2.2 본 프로젝트의 GPT 구조

```python
# model/gpt.py 구조

GPT
├── GPTEmbedding          # 토큰 + 위치 임베딩
│   ├── TokenEmbedding    # 토큰 → 벡터
│   └── PositionalEmbedding # 위치 정보
├── TransformerBlock × 6  # 6개의 Transformer 블록
│   ├── LayerNorm
│   ├── MultiHeadSelfAttention
│   ├── LayerNorm
│   └── FeedForward
├── LayerNorm (final)     # 최종 정규화
└── lm_head               # 출력 프로젝션 (Weight Tying)
```

---

## 3. 임베딩 (Embedding)

### 3.1 토큰 임베딩 (Token Embedding)

각 토큰(단어/서브워드)을 고정 크기의 벡터로 변환합니다.

```python
# model/embedding.py

class TokenEmbedding(nn.Module):
    def __init__(self, config):
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 8000 (어휘 크기)
            embedding_dim=d_model,      # 320 (벡터 차원)
            padding_idx=0               # 패딩 토큰
        )
```

**예시:**
```
"안녕" → 토큰 ID 127 → [0.12, -0.34, 0.56, ...] (320차원 벡터)
```

### 3.2 위치 임베딩 (Positional Embedding)

Transformer는 순서 정보가 없으므로, 위치 정보를 별도로 추가합니다.

**두 가지 방식:**

1. **Sinusoidal (고정)** - 원본 Transformer
   ```
   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   ```

2. **Learnable (학습)** - GPT-2 스타일 (본 프로젝트)
   ```python
   self.embedding = nn.Embedding(max_seq_len, d_model)  # 학습 가능한 파라미터
   ```

### 3.3 최종 임베딩

```python
# 토큰 임베딩 + 위치 임베딩 + 드롭아웃
embeddings = token_emb + pos_emb
embeddings = dropout(embeddings)
```

---

## 4. Self-Attention 메커니즘

### 4.1 Attention이란?

"어떤 정보에 **집중**할지 결정하는 메커니즘"

```
문장: "그 고양이가 매트 위에 앉아 있다. 그것은 귀엽다."

"그것"이 무엇을 가리키는지 알려면?
→ "고양이"에 높은 attention을 부여
```

### 4.2 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**구성 요소:**
- **Q (Query)**: "무엇을 찾고 있는가?"
- **K (Key)**: "무엇을 제공할 수 있는가?"
- **V (Value)**: "실제 정보"

```python
# model/attention.py

# 1. Q, K, V 계산
qkv = self.qkv_proj(x)  # [batch, seq, 3*d_model]
q, k, v = qkv.split(d_model, dim=-1)

# 2. Attention Score 계산
attn_scores = torch.matmul(q, k.transpose(-2, -1))  # QK^T
attn_scores = attn_scores / math.sqrt(head_dim)     # Scale by √d_k

# 3. Softmax로 정규화
attn_probs = F.softmax(attn_scores, dim=-1)

# 4. Value와 곱하기
output = torch.matmul(attn_probs, v)
```

### 4.3 Multi-Head Attention

여러 개의 Attention을 병렬로 수행하여 다양한 관계를 학습합니다.

```
┌────────────────────────────────────────────────────┐
│              Multi-Head Attention                  │
│                                                    │
│   Head 1: 문법적 관계 학습                          │
│   Head 2: 의미적 관계 학습                          │
│   Head 3: 위치 관계 학습                            │
│   ...                                              │
│   Head 8: 기타 패턴 학습                            │
│                                                    │
│   → Concat → Linear → Output                       │
└────────────────────────────────────────────────────┘
```

**본 프로젝트 설정:**
- `n_heads = 8`
- `d_model = 320`
- `head_dim = 320 / 8 = 40`

### 4.4 Causal Mask (인과적 마스킹)

GPT는 미래 토큰을 보면 안 되므로, **Causal Mask**를 적용합니다.

```
Attention Matrix (마스킹 전):
     t1   t2   t3   t4
t1  0.3  0.2  0.3  0.2
t2  0.1  0.4  0.3  0.2
t3  0.2  0.2  0.4  0.2
t4  0.1  0.3  0.2  0.4

Causal Mask 적용 후:
     t1   t2   t3   t4
t1  0.3  -∞   -∞   -∞   ← t1은 자기 자신만 봄
t2  0.1  0.4  -∞   -∞   ← t2는 t1, t2만 봄
t3  0.2  0.2  0.4  -∞   ← t3는 t1, t2, t3만 봄
t4  0.1  0.3  0.2  0.4  ← t4는 모두 봄
```

```python
# 상삼각 행렬로 마스크 생성
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
```

---

## 5. Feed-Forward Network

각 토큰 위치에 독립적으로 적용되는 2층 신경망입니다.

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

```python
# model/feedforward.py

class FeedForward(nn.Module):
    def __init__(self, config):
        self.fc1 = nn.Linear(d_model, d_ff)   # 320 → 1280 (확장)
        self.fc2 = nn.Linear(d_ff, d_model)   # 1280 → 320 (축소)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)      # 확장
        x = self.gelu(x)     # 활성화
        x = self.fc2(x)      # 축소
        return x
```

### GELU vs ReLU

| 활성화 함수 | 특징 |
|-------------|------|
| ReLU | `max(0, x)` - 단순, 빠름 |
| GELU | `x * Φ(x)` - 부드러운 곡선, GPT/BERT에서 사용 |

```
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

---

## 6. Transformer Block

### 6.1 Pre-LayerNorm vs Post-LayerNorm

**Post-LayerNorm (원본 Transformer):**
```
x → Attention → Add → LayerNorm → FFN → Add → LayerNorm
```

**Pre-LayerNorm (GPT-2, 본 프로젝트):**
```
x → LayerNorm → Attention → Add → LayerNorm → FFN → Add
```

Pre-LayerNorm이 학습 안정성이 더 좋아 대부분의 현대 LLM에서 사용됩니다.

```python
# model/transformer_block.py

def forward(self, x):
    # Self-Attention with Pre-LayerNorm
    residual = x
    x = self.ln1(x)           # LayerNorm 먼저
    x = self.attention(x)
    x = x + residual          # Residual Connection

    # FFN with Pre-LayerNorm
    residual = x
    x = self.ln2(x)           # LayerNorm 먼저
    x = self.feedforward(x)
    x = x + residual          # Residual Connection

    return x
```

### 6.2 Residual Connection

깊은 네트워크에서 그래디언트 소실을 방지합니다.

```
출력 = 입력 + 서브레이어(입력)
```

이를 통해 그래디언트가 직접 흐를 수 있는 "고속도로"를 만듭니다.

### 6.3 Layer Normalization

각 샘플 내에서 정규화하여 학습을 안정화합니다.

```
LayerNorm(x) = γ * (x - μ) / (σ + ε) + β

μ = mean(x)
σ = std(x)
γ, β = 학습 가능한 파라미터
```

---

## 7. 학습 (Training)

### 7.1 Language Modeling Objective

다음 토큰 예측 (Next Token Prediction):

```
입력:  [BOS] 오늘  날씨가  좋다
레이블:      오늘  날씨가  좋다  [EOS]

Loss = -Σ log P(토큰_t | 토큰_1, ..., 토큰_{t-1})
```

```python
# model/gpt.py

# Shift for next-token prediction
shift_logits = logits[:, :-1, :]  # 마지막 제외
shift_labels = labels[:, 1:]      # 첫번째 제외

loss = F.cross_entropy(shift_logits, shift_labels)
```

### 7.2 Weight Tying

입력 임베딩과 출력 프로젝션의 가중치를 공유합니다.

```python
# 파라미터 절약 + 성능 향상
self.lm_head.weight = self.embedding.token_embedding.embedding.weight
```

**효과:**
- 파라미터 수 감소 (vocab_size × d_model 절약)
- 의미적 일관성 향상

### 7.3 Optimizer: AdamW

Adam + Weight Decay (L2 정규화)

```python
optimizer = AdamW(
    params,
    lr=3e-4,
    betas=(0.9, 0.95),    # β1, β2
    weight_decay=0.01      # L2 정규화
)
```

**Weight Decay 제외 대상:**
- Bias
- LayerNorm의 γ, β

```python
# training/trainer.py

for name, param in model.named_parameters():
    if "bias" in name or "ln" in name:
        no_decay_params.append(param)  # weight decay 제외
    else:
        decay_params.append(param)
```

### 7.4 Learning Rate Schedule

**Warmup + Cosine Decay:**

```
        lr
        ↑
   max  │    ╱╲
        │   ╱  ╲
        │  ╱    ╲
        │ ╱      ╲╲
   min  │╱         ╲╲___
        └──────────────────→ steps
          warmup   decay
```

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps  # Linear warmup

    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + cos(π * progress))
```

### 7.5 Gradient Clipping

그래디언트 폭발을 방지합니다.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 8. 텍스트 생성 (Inference)

### 8.1 Greedy Decoding

항상 가장 확률 높은 토큰 선택:

```python
next_token = torch.argmax(logits, dim=-1)
```

**단점:** 반복적이고 지루한 텍스트 생성

### 8.2 Temperature Sampling

확률 분포를 조절합니다.

```python
logits = logits / temperature
probs = softmax(logits)
next_token = multinomial(probs)
```

| Temperature | 효과 |
|-------------|------|
| T < 1.0 | 더 확정적 (집중) |
| T = 1.0 | 원래 분포 |
| T > 1.0 | 더 랜덤 (다양) |

```
T=0.5: [0.7, 0.2, 0.1] → [0.9, 0.08, 0.02]  # 더 집중
T=2.0: [0.7, 0.2, 0.1] → [0.45, 0.35, 0.2]  # 더 분산
```

### 8.3 Top-K Sampling

상위 K개 토큰만 고려합니다.

```python
# Top-50 예시
top_k_logits = torch.topk(logits, k=50)
# 나머지 토큰은 -inf로 마스킹
```

### 8.4 Top-P (Nucleus) Sampling

누적 확률이 P를 넘을 때까지의 토큰만 고려합니다.

```python
sorted_probs = sort(probs, descending=True)
cumsum = cumulative_sum(sorted_probs)

# 누적 확률 > 0.9인 토큰 제외
mask = cumsum > 0.9
```

**예시 (Top-P = 0.9):**
```
토큰:    A     B     C     D     E
확률:   0.5   0.3   0.15  0.04  0.01
누적:   0.5   0.8   0.95  0.99  1.0
              ↑
         여기까지만 샘플링 (A, B, C)
```

### 8.5 일반적인 설정 조합

```python
# 창의적인 텍스트
temperature=1.0, top_k=50, top_p=0.95

# 일관된 텍스트
temperature=0.7, top_k=40, top_p=0.9

# 매우 안전한 텍스트
temperature=0.3, top_k=10, top_p=0.8
```

---

## 9. 토크나이저 (Tokenizer)

### 9.1 서브워드 토크나이징

단어를 더 작은 단위로 분할합니다.

```
"안녕하세요" → ["안녕", "하세요"] 또는 ["안", "녕", "하세요"]
```

**장점:**
- OOV(Out-of-Vocabulary) 문제 해결
- 어휘 크기 감소
- 희귀 단어도 처리 가능

### 9.2 BPE (Byte Pair Encoding)

가장 빈번한 문자/토큰 쌍을 반복적으로 병합합니다.

```
초기: ['l', 'o', 'w', '</w>'] + ['l', 'o', 'w', 'e', 'r', '</w>'] + ...

1단계: 'lo'가 가장 빈번 → ['lo', 'w', '</w>'] + ['lo', 'w', 'e', 'r', '</w>']
2단계: 'low'가 가장 빈번 → ['low', '</w>'] + ['low', 'e', 'r', '</w>']
...
```

### 9.3 SentencePiece

언어에 독립적인 서브워드 토크나이저입니다.

```python
# tokenizer/trainer.py

spm.SentencePieceTrainer.train(
    input=text_file,
    model_type="bpe",
    vocab_size=8000,
    character_coverage=0.9995,  # 한국어는 높게 설정
    pad_id=0, bos_id=1, eos_id=2, unk_id=3
)
```

### 9.4 특수 토큰

| 토큰 | ID | 용도 |
|------|-----|------|
| `<pad>` | 0 | 패딩 (길이 맞추기) |
| `<s>` | 1 | 문장 시작 (BOS) |
| `</s>` | 2 | 문장 끝 (EOS) |
| `<unk>` | 3 | 미등록 토큰 |

---

## 10. 주요 하이퍼파라미터

### 10.1 모델 하이퍼파라미터

| 파라미터 | 본 프로젝트 | 설명 |
|----------|-------------|------|
| `vocab_size` | 8,000 | 어휘 크기 |
| `max_seq_len` | 256 | 최대 시퀀스 길이 |
| `d_model` | 320 | 히든 차원 |
| `n_heads` | 8 | Attention 헤드 수 |
| `n_layers` | 6 | Transformer 블록 수 |
| `d_ff` | 1,280 | FFN 중간 차원 |
| `dropout` | 0.1 | 드롭아웃 비율 |

### 10.2 스케일링 법칙

모델 크기와 성능의 관계:

```
Loss ∝ N^(-0.076)  (N: 파라미터 수)
Loss ∝ D^(-0.095)  (D: 데이터 크기)
Loss ∝ C^(-0.050)  (C: 계산량)
```

**일반적인 비율:**
- `d_ff ≈ 4 × d_model`
- `head_dim = d_model / n_heads` (보통 64~128)

### 10.3 학습 하이퍼파라미터

| 파라미터 | 본 프로젝트 | 범위 |
|----------|-------------|------|
| `learning_rate` | 3e-4 | 1e-5 ~ 1e-3 |
| `batch_size` | 32 | 모델/GPU에 따라 |
| `warmup_steps` | 1,000 | 전체의 1~10% |
| `weight_decay` | 0.01 | 0.01 ~ 0.1 |
| `max_grad_norm` | 1.0 | 0.5 ~ 2.0 |

---

## 11. 추가 학습 자료

### 11.1 필수 논문

1. **Attention Is All You Need** (2017)
   - Transformer 원본 논문
   - https://arxiv.org/abs/1706.03762

2. **GPT-2: Language Models are Unsupervised Multitask Learners** (2019)
   - GPT-2 아키텍처
   - https://openai.com/research/better-language-models

3. **GPT-3: Language Models are Few-Shot Learners** (2020)
   - 스케일링 법칙, In-context Learning
   - https://arxiv.org/abs/2005.14165

4. **LLaMA: Open and Efficient Foundation Language Models** (2023)
   - 효율적인 LLM 설계
   - https://arxiv.org/abs/2302.13971

### 11.2 추천 강의

- **Stanford CS224N**: Natural Language Processing with Deep Learning
- **Andrej Karpathy의 nanoGPT**: 처음부터 GPT 구현
- **Jay Alammar의 Illustrated Transformer**: 시각적 설명

### 11.3 실습 코드베이스

| 저장소 | 설명 |
|--------|------|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 가장 단순한 GPT 구현 |
| [minGPT](https://github.com/karpathy/minGPT) | 교육용 GPT 구현 |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | C++로 구현된 LLaMA |
| [transformers](https://github.com/huggingface/transformers) | HuggingFace 라이브러리 |

### 11.4 핵심 개념 정리

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM 학습 로드맵                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 기초                                                        │
│     └── PyTorch 기본 → 신경망 이해 → 역전파                      │
│                                                                 │
│  2. NLP 기초                                                    │
│     └── 토크나이징 → 임베딩 → RNN/LSTM 이해                      │
│                                                                 │
│  3. Transformer                                                 │
│     └── Self-Attention → Multi-Head → 전체 구조                 │
│                                                                 │
│  4. GPT 구현                                                    │
│     └── Causal Mask → Pre-LayerNorm → 텍스트 생성               │
│                                                                 │
│  5. 학습 기법                                                   │
│     └── AdamW → LR Schedule → Gradient Clipping                │
│                                                                 │
│  6. 고급 주제                                                   │
│     └── 스케일링 → RLHF → 효율적 추론                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 부록: 파라미터 계산

본 프로젝트 모델의 파라미터 수 계산:

```
1. Token Embedding: vocab_size × d_model
   = 8,000 × 320 = 2,560,000

2. Position Embedding: max_seq_len × d_model
   = 256 × 320 = 81,920

3. 각 Transformer Block:
   - QKV Projection: d_model × 3 × d_model = 320 × 960 = 307,200
   - Output Projection: d_model × d_model = 320 × 320 = 102,400
   - FFN fc1: d_model × d_ff = 320 × 1,280 = 409,600
   - FFN fc2: d_ff × d_model = 1,280 × 320 = 409,600
   - LayerNorm × 2: 2 × 2 × d_model = 1,280
   - Block 총계: 1,230,080

4. 6개 Block: 6 × 1,230,080 = 7,380,480

5. Final LayerNorm: 2 × d_model = 640

6. LM Head: Weight Tying으로 0 (임베딩과 공유)

총계: 2,560,000 + 81,920 + 7,380,480 + 640 ≈ 10,023,040
```

---

*이 문서는 한국어 GPT 구현 프로젝트의 학습 가이드입니다.*
