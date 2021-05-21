#!/bin/sh

NAME="no-sampling-cifar10-lr-1e-4-kl-sweep"

screen -dmS "no-sampling-logging" bash -c "source /home/ananya/env/bin/activate; tensorboard dev upload --logdir lightning_logs/ --name $NAME; exec sh"