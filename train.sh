#!/bin/bash

# Get the worker index from the environment variable
WORKER_INDEX=$PMI_RANK

# Set the TF_CONFIG environment variable based on the worker index
export TF_CONFIG=$(cat ./TF_CONFIG_${WORKER_INDEX}.json)

# Run the training script
python train.py
