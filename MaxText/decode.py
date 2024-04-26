# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=assets/tokenizer.llama3 load_parameters_path=gs://maxtext-llama/test/2024-04-24-04-28/decode-ckpt-maxtext/0/items per_device_batch_size=1 run_name=runner_$(date +%Y-%m-%d-%H-%M) max_prefill_predict_length=8 max_target_length=16 dataset_type=synthetic steps=10 async_checkpointing=false model_name=llama3-8b attention=dot_product prompt='I love to' decode_sampling_strategy=nucleus decode_sampling_nucleus_p=0.9 decode_sampling_temperature=0.6
"""

"""CLI Utility for Running Inference on a Single Stream"""

import jax
import jax.numpy as jnp
import maxengine
from jetstream.engine import token_utils
from tokenizer import Tokenizer

import os
import pyconfig
import sys


def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()

  text = config.prompt
  # metadata = engine.get_tokenizer()
  # vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  # tokenizer = vocab.tokenizer
  # tokens, true_length = token_utils.tokenize_and_pad(
  #     text, vocab, is_bos=True, prefill_lengths=[config.max_prefill_predict_length]
  # )
  tokenizer = Tokenizer(model_path=config.tokenizer_path)
  tokens = tokenizer.encode(text, bos=True, eos=False)
  true_length = len(tokens)
  pad_id = tokenizer.pad_id
  tokens.extend([pad_id] * (config.max_prefill_predict_length - len(tokens)))
  tokens = jnp.array(tokens)
  assert tokens.size <= config.max_prefill_predict_length, "can't take too many tokens"
  assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
  prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
  slot = 0

  decode_state = engine.init_decode_state()
  decode_state = engine.insert(prefill_result, decode_state, slot=slot)

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  for _ in steps:
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    sampled_tokens_list.append(sampled_tokens)

  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  output = tokenizer.decode(results)
  print(f"Input `{text}` -> `{output}`")

  if config.autoregressive_decode_assert != "":
    assert (
        output == config.autoregressive_decode_assert
    ), f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  validate_config(cfg)
  main(cfg)
