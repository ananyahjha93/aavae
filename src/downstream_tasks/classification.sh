# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/downstream_tasks/classification.py --dataset places205 --data_path /home/ananya/places205/images256 --seed $(date +%s) --ckpt_path /home/ananya/run_imagenet/aavae/lightning_logs/aavae/epoch=2099-step=5048399.ckpt --batch_size 64 --num_workers 6 --gpus 4
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/downstream_tasks/classification.py --dataset places205 --data_path /home/ananya/places205/images256 --seed $(date +%s) --ckpt_path /home/ananya/run_imagenet/aavae/lightning_logs/vae/epoch=1699-step=4086799.ckpt --batch_size 64 --num_workers 6 --gpus 4