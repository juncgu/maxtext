"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Provides op for tokenizing a dataset."""

from typing import Any, Dict, Iterable

import dataclasses
from absl import logging
from pathlib import Path
import tensorflow as tf
import tensorflow_text as tftxt
import tiktoken
from tiktoken.load import load_tiktoken_bpe

import max_logging


Features = Dict[str, tf.Tensor]


def _load_sentencepiece_tokenizer(tokenizer_path: str, add_bos: bool = False, add_eos: bool = True, reverse: bool = False):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  max_logging.log(f"Tokenizer path: {tokenizer_path}")
  with tf.io.gfile.GFile(tokenizer_path, "rb") as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
  return sp_tokenizer


def load_sentencepiece_tokenizer(tokenizer_path: str, add_bos=False, add_eos=True):
  """Loads the sentencepiece tokenizer at `tokenizer_path` or trains a one from `dataset`."""
  try:
    sp_tokenizer = _load_sentencepiece_tokenizer(tokenizer_path, add_bos, add_eos)
    return sp_tokenizer
  except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError):
    logging.info("SentencePiece vocab not found, Run train_tokenizer.py")
    return None
 
class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special = set(),
        disallowed_special = (),
    ):
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str

        t = self.model.encode(
                    s,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
        )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(t)


@dataclasses.dataclass
class TokenizeOp:
  sp_tokenizer: Any
  data_keys: Iterable[str] = ("inputs", "targets")

  def __call__(self, features: Features) -> Features:
    for k in self.data_keys:
      features[k] = self.sp_tokenizer.tokenize(features[k])
    return features
