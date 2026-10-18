"""Step 4: train the tiny GPT to predict the next character of Naruto dialogue."""
import time, torch
from tokenizer import text, encode, decode, vocab_size
from model import TinyGPT

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"
batch_size, block_size = 32, 128
max_iters, eval_every, lr = 5000, 250, 3e-4

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]   # 10% held out to check it's not just memorising

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])  # targets = input shifted by one
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        out[split] = torch.stack([model(*get_batch(split))[1] for _ in range(50)]).mean().item()
    model.train()
    return out

model = TinyGPT(vocab_size, block_size=block_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
print(f"training on {device}, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params", flush=True)

t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_every == 0:
        l = estimate_loss()
        print(f"step {it:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} | {time.time()-t0:.0f}s", flush=True)
    x, y = get_batch("train")
    _, loss = model(x, y)          # 1. guess the next characters, measure how wrong
    opt.zero_grad(set_to_none=True)
    loss.backward()                # 2. work out how each parameter caused the error
    opt.step()                     # 3. nudge every parameter to be a bit less wrong

torch.save(model.state_dict(), "naruto_gpt.pt")
print("saved naruto_gpt.pt\n--- sample ---")
start = torch.tensor([encode("\n")], device=device)
print(decode(model.generate(start, 500)[0].tolist()))
