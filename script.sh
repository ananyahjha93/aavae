#!/bin/sh

# can set min, max and n
LOG_SCALE=0
LR=1e-4
KL=(-1 -1 -1 -1 1e-3 1e-2 0.1 1)

for i in {4..7}; do
    screen -dmS "run$i" bash -c "source /home/ananya/env/bin/activate; python setup.py install; CUDA_VISIBLE_DEVICES=$i python src/vae.py --seed $(date +%s) --denoising --gpus 1 --batch_size 256 --warmup_epochs 10 --val_samples 16 --weight_decay 0 --learning_rate $LR --kl_coeff ${KL[$i]} --log_scale $LOG_SCALE; exec sh"
    sleep 5
done
