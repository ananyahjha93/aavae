# VAE

grid train --grid_instance_type p3.2xlarge --grid_gpus 1 --grid_name cifar10-kl-lr-wd --g_use_spot src/vae.py --gpus 1 --denoising --val_samples 16 --online_ft --batch_size 256 --kl_coeff "[0.001, 0.01, 0.1, 1, 10]" --weight_decay "[0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]" --learning_rate "[1e-4, 5e-4, 1e-3]"


## Sweep commands
```bash
grid run --g_instance_type 8_v100_16gb \
         --g_gpus 1 \
         --grid_name vae-cifar10-baseline-v1 \
         --g_use_spot \
         --g_strategy random_search \
         --g_trials 128 \
         vae.py \
         --gpus 1 \
         --denoising \
         --online_ft \
         --batch_size 256 \
         -kl_coeff "[1e-3, 1e-2, 1e-1, 1, 10]" \
         --weight_decay "[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]" \
         --learning_rate "[1e-4, 5e-4, 1e-3]" \
         --val_samples "[1, 8, 16, 32]"
```