"""Step 1: turn raw .srt subtitle files into one clean text file (input.txt)."""
import re
from pathlib import Path

lines = []
for path in sorted(Path("data").glob("*.srt")):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    for block in re.split(r"\n\s*\n", raw.replace("\r", "")):
        rows = block.strip().split("\n")
        # drop the cue number and the "00:00:30,449 --> ..." timestamp rows
        text = [r for r in rows if not r.strip().isdigit() and "-->" not in r]
        text = " ".join(text)
        text = re.sub(r"<[^>]+>|\{[^}]+\}", "", text)   # strip <font> and {\an8} tags
        text = re.sub(r"\s+", " ", text).strip()
        if text and not re.search(r"subtitle|opensubtitles|www\.|http", text, re.I):
            lines.append(text)

corpus = "\n".join(lines)
Path("input.txt").write_text(corpus, encoding="utf-8")
print(f"{len(lines):,} lines, {len(corpus):,} characters, {len(set(corpus))} unique characters")
