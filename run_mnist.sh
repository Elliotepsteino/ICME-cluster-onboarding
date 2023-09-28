#!/bin/bash

# Define different LR values for each script
LR_1=0.1
LR_2=0.5

# Launch the first training script on GPU 0 with LR_1 and epochs set to 3
CUDA_VISIBLE_DEVICES=0 python mnist_pytorch_example.py --lr $LR_1 --epochs 3 &

# Launch the second training script on GPU 1 with LR_2 and epochs set to 3
CUDA_VISIBLE_DEVICES=1 python mnist_pytorch_example.py --lr $LR_2 --epochs 3 &