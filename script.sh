#!/bin/sh

python src/vae.py --dataset imagenet --data_path /home/ananya/imagenet/data --seed $(date +%s) --ckpt_path /home/ananya/run_imagenet/aavae/lightning_logs/version_0/checkpoints/epoch=599-step=1442399.ckpt --online_ft --gpus 8 --max_epochs 3200 --val_samples 1 --kl_coeff 0 --log_scale 0 --num_workers 8 --batch_size 64 --warmup_epochs 10 --learning_rate 5e-4
