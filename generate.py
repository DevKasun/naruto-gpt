"""Step 5: use the trained model. Usage: uv run generate.py "Sasuke!" [length] [temperature]"""
import sys, torch
from tokenizer import encode, decode, vocab_size
from model import TinyGPT

prompt = sys.argv[1] if len(sys.argv) > 1 else "Naruto!"
length = int(sys.argv[2]) if len(sys.argv) > 2 else 300
temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8   # lower = safer, higher = wilder

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = TinyGPT(vocab_size).to(device)
model.load_state_dict(torch.load("naruto_gpt.pt", map_location=device))
model.eval()

idx = torch.tensor([encode(prompt + "\n")], device=device)
print(decode(model.generate(idx, length, temperature)[0].tolist()))
