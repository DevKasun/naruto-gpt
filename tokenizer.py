"""Step 2: a character-level tokenizer. Each unique character gets an integer id."""
from pathlib import Path

text = Path("input.txt").read_text(encoding="utf-8")
chars = sorted(set(text))                      # the vocabulary
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # string -> id
itos = {i: ch for i, ch in enumerate(chars)}   # id -> string

def encode(s): return [stoi[c] for c in s]
def decode(ids): return "".join(itos[i] for i in ids)

if __name__ == "__main__":
    print("vocab size:", vocab_size)
    ids = encode("Believe it!")
    print("encode('Believe it!') ->", ids)
    print("decode back ->", decode(ids))