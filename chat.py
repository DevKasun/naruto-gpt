"""Chat with your Naruto-tuned Qwen. Usage: uv run chat.py"""
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

SYSTEM = "You write dialogue in the style of Naruto Shippuden. Keep lines short and dramatic, like anime subtitles."
model, tok = load("Qwen/Qwen2.5-1.5B-Instruct", adapter_path="adapters")
sampler = make_sampler(temp=0.8, top_p=0.9)
procs = make_logits_processors(repetition_penalty=1.3, repetition_context_size=64)

print("Naruto GPT is ready! Type a scene or line (Ctrl+C to quit).\n")
while True:
    try:
        user = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nSee ya, dattebayo!"); break
    if not user: continue
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Continue this Naruto Shippuden scene: {user}"}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    print("Naruto GPT:")
    for r in stream_generate(model, tok, prompt, max_tokens=200, sampler=sampler, logits_processors=procs):
        print(r.text, end="", flush=True)
    print("\n")
