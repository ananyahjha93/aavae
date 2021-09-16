#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/vae.py --dataset imagenet --data_path /home/ananya/imagenet/data/imagenet_2012 --seed $(date +%s) --ckpt_path /home/ananya/run_imagenet/aavae/lightning_logs/version_4/checkpoints/epoch=1699-step=4086799.ckpt --online_ft --denoising --gpus 4 --max_epochs 3200 --val_samples 1 --kl_coeff 0 --log_scale 0 --num_workers 16 --batch_size 128 --warmup_epochs 10 --learning_rate 5e-4

# CUDA_VISIBLE_DEVICES=4,5 python src/vae.py --dataset imagenet --data_path /home/ananya/imagenet/data/imagenet_2012 --seed $(date +%s) --online_ft --denoising --gpus 2 --max_epochs 1600 --val_samples 1 --kl_coeff 0 --log_scale 0 --num_workers 16 --batch_size 128 --warmup_epochs 5 --learning_rate 2.5e-4

# CUDA_VISIBLE_DEVICES=6 python src/vae.py --dataset imagenet --data_path /home/ananya/imagenet/data/imagenet_2012 --seed $(date +%s) --online_ft --denoising --gpus 1 --max_epochs 800 --val_samples 1 --kl_coeff 0 --log_scale 0 --num_workers 16 --batch_size 128 --warmup_epochs 3 --learning_rate 1e-4

# CUDA_VISIBLE_DEVICES=7 python src/vae.py --dataset imagenet --data_path /home/ananya/imagenet/data/imagenet_2012 --seed $(date +%s) --online_ft --denoising --gpus 1 --max_epochs 800 --val_samples 1 --kl_coeff 0 --log_scale 0 --num_workers 16 --batch_size 128 --warmup_epochs 3 --learning_rate 5e-4
