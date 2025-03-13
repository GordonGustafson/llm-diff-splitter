#!/usr/bin/env bash

set -euo pipefail

nvidia-smi

git clone https://github.com/GordonGustafson/llm-diff-splitter.git
cd llm-diff-splitter
pip install -r <(grep -v 'torch==' requirements.txt | grep -v 'torchvision==')

time bash data/clone_repos.sh
time bash data/save_patches.sh
time python data/convert_patches_to_parquet.py

huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-1B

time python train.py

