# Korean GPT

처음부터 구현한 한국어 GPT 언어 모델 (~10M parameters).

Decoder-only Transformer 아키텍처를 사용하며, 한국어 코퍼스(NSMC, KcBERT)로 학습합니다.

## 아키텍처

| 항목 | 값 |
|------|-----|
| 모델 | Decoder-only Transformer (GPT) |
| 파라미터 | ~10M |
| vocab_size | 8,000 |
| d_model | 320 |
| n_heads | 8 |
| n_layers | 6 |
| d_ff | 1,280 |
| max_seq_len | 256 |
| Tokenizer | SentencePiece BPE |

## 프로젝트 구조

```
├── config/          # 모델, 학습, 토크나이저 설정
├── model/           # GPT 모델 (Attention, FFN, Embedding, Transformer Block)
├── tokenizer/       # SentencePiece BPE 토크나이저
├── data/            # Korpora 기반 데이터셋 및 DataLoader
├── training/        # 학습 루프 (Trainer)
├── inference/       # 텍스트 생성 (Top-K, Top-P, Greedy)
├── scripts/         # 실행 스크립트
└── docs/            # 학습 가이드
```

## 설치

```bash
pip install -r requirements.txt
```

### 코퍼스 다운로드

```python
from Korpora import Korpora
Korpora.fetch("nsmc")
Korpora.fetch("kcbert")
```

## 사용법

### 1. 토크나이저 학습

```bash
python scripts/train_tokenizer.py
```

### 2. 모델 학습

```bash
python scripts/train_model.py
```

주요 옵션:

```bash
python scripts/train_model.py \
  --d-model 320 \
  --n-layers 6 \
  --n-heads 8 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --max-epochs 10 \
  --device mps  # cuda, cpu
```

### 3. 텍스트 생성

```bash
# 단일 생성
python scripts/generate.py --prompt "오늘 날씨가 정말"

# 인터랙티브 모드
python scripts/generate.py --interactive

# 예제 프롬프트 실행
python scripts/generate.py --examples
```

생성 옵션:

```bash
python scripts/generate.py \
  --prompt "옛날 옛적에" \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9 \
  --max-length 100
```

## 요구사항

- Python 3.10+
- PyTorch >= 2.0.0
- SentencePiece >= 0.1.99
- Korpora >= 0.2.0

## 주요 구현 사항

- **Weight Tying**: 임베딩과 출력 레이어 가중치 공유
- **Causal Attention Mask**: 자기회귀 생성을 위한 마스킹
- **Sliding Window Dataset**: 긴 텍스트를 겹치는 윈도우로 분할
- **Warmup + Cosine Decay**: 학습률 스케줄링
- **Top-K / Top-P Sampling**: 다양한 디코딩 전략 지원
- **MPS 최적화**: Apple Silicon 지원
