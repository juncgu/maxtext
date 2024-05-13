#!/bin/bash
set -e
batch_sizes=(16 32 48 64 72 96)
prefill_cache_lengths=(1024 2048)
generate_cache_lengths=(1024 2048 3072 4096 5120 6144 7168)
model_name="gemma-2b"
generate_steps=1000
LOAD_PARAMETERS_PATH="gs://sysres-disagg-ml-maxtext/runner/gemma-2b_unscanned_chkpt_0507/checkpoints/0/items"
log_dir="${PWD}/step${generate_steps}-microbenchmark"

mkdir ${log_dir}
for batch_size in "${batch_sizes[@]}"; do
  for prefill_cache_length in "${prefill_cache_lengths[@]}"; do
    for generate_cache_length in "${generate_cache_lengths[@]}"; do
      target_cache_length=$(($generate_cache_length + $prefill_cache_length))
      if [ ${target_cache_length} -le 8192 ]; then
        echo ${target_cache_length}
        exp_name="${model_name}_bs${batch_size}_p${prefill_cache_length}_g${generate_cache_length}_step${generate_steps}"
        echo "python3 MaxText/inference_microbenchmark.py MaxText/configs/base.yml \
                tokenizer_path=assets/tokenizer.gemma weight_dtype=bfloat16 \
                model_name=${model_name} ici_fsdp_parallelism=1 ici_data_parallelism=1 ici_autoregressive_parallelism=-1 \
                scan_layers=false \
                max_prefill_predict_length=${prefill_cache_length} \
                max_target_length=${target_cache_length} \
                per_device_batch_size=${batch_size} \
                load_parameters_path=${LOAD_PARAMETERS_PATH} generate_length=${generate_steps} \
                output_dir=${log_dir} \
                run_name=${exp_name}" | tee -a ${log_dir}/${exp_name}.log ${log_dir}/microbenchmark_summary.log
        echo "======================================================================="
        python3 MaxText/inference_microbenchmark.py MaxText/configs/base.yml \
                tokenizer_path=assets/tokenizer.gemma weight_dtype=bfloat16 \
                model_name=${model_name} ici_fsdp_parallelism=1 ici_data_parallelism=1 ici_autoregressive_parallelism=-1 \
                scan_layers=false \
                max_prefill_predict_length=${prefill_cache_length} \
                max_target_length=${target_cache_length} \
                per_device_batch_size=${batch_size} \
                load_parameters_path=${LOAD_PARAMETERS_PATH} generate_length=${generate_steps} \
                output_dir=${log_dir} \
                run_name=${exp_name} >> ${log_dir}/microbenchmark_summary.log || true

        sleep 2s
      fi
    done
  done
done
