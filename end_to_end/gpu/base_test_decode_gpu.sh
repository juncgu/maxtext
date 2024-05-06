#!/bin/bash

# This script provides a convenient set of default configs used for testing decode.py on GPU

RUN_NAME=decode-$(date +%Y-%m-%d-%H-%M)-$RANDOM
BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
DATASET_PATH=gs://maxtext-dataset
STEPS=2 
ENABLE_CHECKPOINTING=false
ATTENTION=dot_product
MAX_TARGET_LENGTH=128
PER_DEVICE_BATCH_SIZE=1

python3 MaxText/decode.py MaxText/configs/base.yml\
    run_name=${RUN_NAME}\
    base_output_directory=${BASE_OUTPUT_DIRECTORY}\
    dataset_path=${DATASET_PATH}\
    steps=${STEPS}\
    enable_checkpointing=${ENABLE_CHECKPOINTING}\
    attention=${ATTENTION}\
    max_target_length=${MAX_TARGET_LENGTH}\
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE}\
    $@
