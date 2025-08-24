#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2 accelerate launch --config_file configs/accelerate/2gpu.yaml scripts/train.py --config configs/sra_wikitext103-ablation-ffn-parity.yaml
