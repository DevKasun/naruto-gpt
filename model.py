"""Step 3: a tiny GPT (decoder-only transformer), the same idea behind ChatGPT, just small."""
import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """One head of self-attention: each character looks back at earlier characters
    and decides which ones matter for predicting what comes next."""
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)    # "what I contain"
        self.query = nn.Linear(n_embd, head_size, bias=False)  # "what I'm looking for"
        self.value = nn.Linear(n_embd, head_size, bias=False)  # "what I share if picked"
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # similarity scores
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # no peeking ahead
        att = self.dropout(F.softmax(att, dim=-1))
        return att @ self.value(x)

class MultiHead(nn.Module):
    """Several attention heads in parallel, each free to notice different patterns."""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, hs, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class Block(nn.Module):
    """One transformer block: attention (talk to other characters) then a
    feed-forward network (think about what was heard)."""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.attn = MultiHead(n_embd, n_head, block_size, dropout)
        self.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # "+ x" is a residual connection
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embd=192, n_head=6, n_layer=6, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # character id -> vector
        self.pos_emb = nn.Embedding(block_size, n_embd)   # position -> vector
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)         # vector -> score for each next char

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.blocks(self.tok_emb(idx) + self.pos_emb(pos))
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)  # sample next char
        return idx

if __name__ == "__main__":
    from tokenizer import vocab_size
    m = TinyGPT(vocab_size)
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")
