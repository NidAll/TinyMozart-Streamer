from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from miditok import REMI, TokenizerConfig
from torch.nn import functional as F


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


MODEL_ID = "LH-Tech-AI/TinyMozart_v2_85M"
MODEL_REVISION = "584f8dbcb81e1a47421a066e84f9b7b90857650f"

N_EMBD = 512
N_HEAD = 8
N_LAYER = 8
BLOCK_SIZE = 1024
DROPOUT = 0.3
VOCAB_SIZE = 387


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N_EMBD, dim=2)
        q = q.view(batch, tokens, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch, tokens, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch, tokens, self.num_heads, self.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(batch, tokens, channels))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(N_HEAD, N_EMBD // N_HEAD)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyMozart(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, token_count = idx.shape
        positions = torch.arange(token_count, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)
        x = self.blocks(x)
        return self.lm_head(self.ln_f(x))


@dataclass(frozen=True)
class GenerationSettings:
    chunk_tokens: int = 512
    temperature: float = 0.90
    top_p: float = 0.92
    top_k: int = 34
    repetition_penalty: float = 1.35
    repetition_window: int = 96
    no_repeat_ngram_size: int = 5
    candidate_count: int = 4
    temperature_jitter: float = 0.08
    seed: int | None = None


class TinyMozartGenerator:
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = REMI(
            TokenizerConfig(
                num_velocities=16,
                use_chords=True,
                use_tempos=True,
                use_time_signatures=True,
            )
        )
        self.model = TinyMozart().to(self.device)
        self._load_checkpoint()
        self.model.eval()

    def _load_checkpoint(self) -> None:
        checkpoint_path = Path(
            hf_hub_download(
                MODEL_ID,
                "model.pt",
                revision=MODEL_REVISION,
            )
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict)

    @torch.inference_mode()
    def generate(self, context: list[int], settings: GenerationSettings) -> list[int]:
        if not context:
            context = [0]
        x = torch.tensor([context[-BLOCK_SIZE:]], dtype=torch.long, device=self.device)
        new_tokens: list[int] = []
        generator = None
        if settings.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(settings.seed)

        for _ in range(settings.chunk_tokens):
            x_cond = x[:, -BLOCK_SIZE:]
            logits = self.model(x_cond)[:, -1, :] / settings.temperature

            recent_tokens = x[0, -settings.repetition_window :].tolist()
            for token in set(recent_tokens):
                if logits[0, token] < 0:
                    logits[0, token] *= settings.repetition_penalty
                else:
                    logits[0, token] /= settings.repetition_penalty

            blocked_tokens = _blocked_ngram_tokens(
                x[0].tolist(),
                settings.no_repeat_ngram_size,
            )
            if blocked_tokens:
                logits[0, blocked_tokens] = -float("inf")

            top_k = min(settings.top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float("inf")

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative_probs > settings.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            logits.scatter_(1, sorted_indices[remove].unsqueeze(0), -float("inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            token_id = int(next_token.item())
            new_tokens.append(token_id)
            x = torch.cat((x, next_token), dim=1)

        return new_tokens


def _blocked_ngram_tokens(tokens: list[int], ngram_size: int) -> list[int]:
    if ngram_size <= 1 or len(tokens) < ngram_size - 1:
        return []

    prefix = tuple(tokens[-(ngram_size - 1) :])
    blocked: set[int] = set()
    for start in range(0, len(tokens) - ngram_size + 1):
        ngram = tokens[start : start + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            blocked.add(ngram[-1])
    return list(blocked)
