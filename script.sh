#!/bin/sh

RECON_COEFF=0.6
LR=2.5e-4
KL=0.1
WARMUP_EPOCHS=10

screen -dmS "recon0" bash -c "source /home/ananya/env/bin/activate; python setup.py install; CUDA_VISIBLE_DEVICES=0 python src/vae.py --seed $(date +%s) --denoising --online_ft --gpus 1 --max_epochs 3200 --batch_size 256 --val_samples 16 --weight_decay 0 --log_scale 0 --learning_rate $LR --warmup_epochs $WARMUP_EPOCHS --kl_coeff $KL --recon_coeff $RECON_COEFF; exec sh"
sleep 5

RECON_COEFF=0.8
LR=2.5e-4
KL=0.1
WARMUP_EPOCHS=10

screen -dmS "recon1" bash -c "source /home/ananya/env/bin/activate; python setup.py install; CUDA_VISIBLE_DEVICES=4 python src/vae.py --seed $(date +%s) --denoising --online_ft --gpus 1 --max_epochs 3200 --batch_size 256 --val_samples 16 --weight_decay 0 --log_scale 0 --learning_rate $LR --warmup_epochs $WARMUP_EPOCHS --kl_coeff $KL --recon_coeff $RECON_COEFF; exec sh"
sleep 5

RECON_COEFF=2
LR=2.5e-4
KL=0.1
WARMUP_EPOCHS=10

screen -dmS "recon2" bash -c "source /home/ananya/env/bin/activate; python setup.py install; CUDA_VISIBLE_DEVICES=7 python src/vae.py --seed $(date +%s) --denoising --online_ft --gpus 1 --max_epochs 3200 --batch_size 256 --val_samples 16 --weight_decay 0 --log_scale 0 --learning_rate $LR --warmup_epochs $WARMUP_EPOCHS --kl_coeff $KL --recon_coeff $RECON_COEFF; exec sh"
