import json, random, os
SYS="You write dialogue in the style of Naruto Shippuden. Keep lines short and dramatic, like anime subtitles."
lines=[l.strip() for l in open("input.txt",encoding="utf-8") if l.strip()]
chunks=[];cur=[];n=0
for l in lines:
    cur.append(l);n+=len(l)+1
    if (len(cur)>=8 and n>=600) or len(cur)>=20 or n>=900:
        chunks.append("\n".join(cur));cur=[];n=0
if len(cur)>=4: chunks.append("\n".join(cur))
random.seed(0);random.shuffle(chunks)
k=int(len(chunks)*0.9)
os.makedirs("lora_data",exist_ok=True)
for name,part in [("train",chunks[:k]),("valid",chunks[k:])]:
    with open(f"lora_data/{name}.jsonl","w",encoding="utf-8") as f:
        for c in part:
            f.write(json.dumps({"messages":[{"role":"system","content":SYS},{"role":"user","content":"Continue this Naruto Shippuden scene:"},{"role":"assistant","content":c}]},ensure_ascii=False)+"\n")
    print(name,len(part))
