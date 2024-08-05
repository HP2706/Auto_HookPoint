#! /bin/bash
conda init
source ~/.bashrc
conda create -n dev_env python=3.11
conda activate dev_env
pip install uv
uv pip install hf_transfer
uv pip install -r dev-requirements.txt
conda init
source ~/.bashrc
export HF_HUB_ENABLE_HF_TRANSFER=1