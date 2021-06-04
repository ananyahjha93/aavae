#!/bin/sh

NAME="aavae-cifar10-recon-coeff-baselines"

screen -dmS "recon-logging" bash -c "source /home/ananya/env/bin/activate; tensorboard dev upload --logdir lightning_logs/ --name $NAME; exec sh"