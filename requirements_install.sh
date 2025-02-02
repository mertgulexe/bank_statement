#!/bin/bash

clear
echo "[SYSTEM INFO] Installing the requirements..."
sleep 2
pip install git+https://github.com/huggingface/transformers --no-cache-dir --upgrade
pip install -r requirements.txt --no-cache-dir --upgrade

echo -e "\n[SYSTEM INFO] We also install \"flash attention\" for faster and memory-efficient inference."
sleep 2
echo "[SYSTEM INFO] Installing..."
pip install flash-attn --no-build-isolation --no-cache-dir --upgrade

echo -e "\n[SYSTEM INFO] Done.\n"
sleep 0.5