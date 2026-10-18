#!/bin/zsh
cd $HOME/Developer/naruto-gpt
UV=/opt/homebrew/bin/uv
echo "downloading model..." 
$UV run python -c "from huggingface_hub import snapshot_download as s;print(s('Qwen/Qwen2.5-1.5B-Instruct'))" || { echo EXIT=download_failed; exit 0; }
echo "training..."
$UV run python -m mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data lora_data --batch-size 1 --iters 800 --learning-rate 1e-5 --num-layers 8 --max-seq-length 384 --grad-checkpoint --mask-prompt --steps-per-report 20 --steps-per-eval 200 --save-every 200 --adapter-path adapters
echo EXIT=$?
