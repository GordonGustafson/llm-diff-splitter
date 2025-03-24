#!/usr/bin/env bash

set -euo pipefail

nvidia-smi

echo 'set-option -g history-limit 10000' >> ~/.tmux.conf
echo 'set-window-option -g mode-keys vi' >> ~/.tmux.conf

git clone https://github.com/GordonGustafson/llm-diff-splitter.git
cd llm-diff-splitter
pip install -r <(grep -v 'torch==' requirements.txt | grep -v 'torchvision==')
apt-get install parallel -y

time bash data/clone_repos.sh
time bash data/run_save_patches_in_parallel.sh
time python data/convert_patches_to_parquet.py

huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-1B

time python train.py
time python train_with_rl.py

