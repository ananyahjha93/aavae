#!/bin/sh

CUDA_VISIBLE_DEVICES=4,5,6,7 python src/ae.py --dataset imagenet --data_path /home/ananya/imagenet/data/imagenet_2012 --seed $(date +%s) --online_ft --gpus 4 --max_epochs 3200 --num_workers 16 --batch_size 128 --warmup_epochs 10 --learning_rate 5e-4
