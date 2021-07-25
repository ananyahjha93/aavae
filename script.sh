#!/bin/sh

python src/ae.py --dataset imagenet --data_path /home/ananya/imagenet/data --seed $(date +%s) --online_ft --gpus 8 --max_epochs 3200 --num_workers 8 --batch_size 64 --warmup_epochs 10 --learning_rate 5e-4
