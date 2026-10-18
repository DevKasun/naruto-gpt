# naruto-gpt

A tiny GPT trained from scratch on Naruto Shippuden subtitles, plus a LoRA fine-tune
of Qwen2.5-1.5B that chats in Naruto's voice.

## Setup

```sh
uv sync
```

Requires Python 3.12+. Training uses MPS on Apple Silicon if available, otherwise CPU.

The LoRA part needs MLX, so it's macOS-only.

## Track 1: character-level GPT (PyTorch)

A ~2.7M-parameter decoder-only transformer that predicts the next character.

```sh
uv run prepare.py     # data/*.srt -> input.txt (strips cue numbers, timestamps, tags)
uv run tokenizer.py   # builds the character vocabulary
uv run train.py       # 5000 steps, saves naruto_gpt.pt, prints a sample
uv run generate.py "Sasuke!" 300 0.8
```

`generate.py` takes an optional prompt, character count, and temperature (lower is
safer, higher is wilder).

Training took about 18 minutes on MPS and finished at train loss 1.22 / val loss 1.39.
Output is coherent Naruto-flavored mush — it learns the shape of the dialogue without
understanding it.

## Track 2: LoRA chat model (MLX)

Fine-tunes Qwen2.5-1.5B-Instruct on the same subtitles, grouped into ~600-character
scenes and wrapped in a chat template.

```sh
uv run prepare_chat_data.py   # input.txt -> lora_data/{train,valid}.jsonl
./run_lora.sh                 # downloads the base model, trains, writes to adapters/
uv run chat.py                # interactive chat with the adapter
```

800 iterations over 8 layers, learning rate 1e-5, LoRA rank 8.

**Note:** as of the last training run the adapter is undertrained — the sample output in
`lora_sample.txt` is the base model falling back to refusals rather than Naruto dialogue.
More iterations, a higher learning rate, or grouping scenes into shorter chunks would
help. `run_lora.sh` currently hardcodes the path to `~/Developer/naruto-gpt`; edit line 2
if you clone this elsewhere.

## Files

| File | What it does |
| --- | --- |
| `prepare.py` | `.srt` subtitles → `input.txt` |
| `tokenizer.py` | Character-level encode/decode |
| `model.py` | `TinyGPT`: attention heads, blocks, generation |
| `train.py` | Training loop, 90/10 train/val split |
| `generate.py` | Sample from the trained checkpoint |
| `prepare_chat_data.py` | `input.txt` → LoRA chat JSONL |
| `run_lora.sh` | LoRA training wrapper |
| `chat.py` | Interactive chat REPL |

`data/` holds the 74 `.srt` files and `adapters/` the LoRA weights. Neither is in
`.gitignore` yet, so add them if the repo is going public.
