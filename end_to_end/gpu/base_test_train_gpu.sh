#!/bin/bash

# This script provides a convenient set of default configs used for testing train.py on GPU

RUN_NAME=train-$(date +%Y-%m-%d-%H-%M)-$RANDOM
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
DATASET_PATH=gs://maxtext-dataset
STEPS=2 
ENABLE_CHECKPOINTING=false
ATTENTION=dot_product

python3 MaxText/train.py MaxText/configs/base.yml \
    run_name=${RUN_NAME}\
    base_output_directory=${BASE_OUTPUT_DIRECTORY}\
    dataset_path=${DATASET_PATH}\
    steps=${STEPS}\
    enable_checkpointing=${ENABLE_CHECKPOINTING}\
    attention=${ATTENTION}\
    $@
