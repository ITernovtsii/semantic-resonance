#!/bin/bash

CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file configs/accelerate/1gpu.yaml scripts/train.py --config outputs_wt103-D512-1024-1_2\@128-4-a4-b6-z0-ppl-q16.9x5_10/sra_wikitext103.yaml --resume outputs_wt103-D512-1024-1_2\@128-4-a4-b6-z0-ppl-q16.9x5_10/best_model.pt
