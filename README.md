# VAE

grid train --grid_config config.yml --grid_name "add-custom-name-here" src/train.py --max_epochs 3200 --gpus 8 --batch_size 128 --learning_rate 1e-3 --warmup_epochs 50 --val_samples 16  --learn_scale 0 --log_scale "[-0.01, -0.05, -0.1, -0.5]" --kl_coeff "[0, 0.1]"