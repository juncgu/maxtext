import jax
from typing import Dict
import tempfile
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from sentencepiece import SentencePieceTrainer
import tensorflow as tf
import tensorflow_text as tftxt

model_path = 'assets/tokenizer.llama3'
num_reserved_special_tokens = 256

def _dump_chars_to_textfile(mergeable_ranks, special_tokens):
  """Write part of a TFDS sentence dataset to lines in a text file.
  Args:
    dataset: tf.dataset containing string-data.
    maxchars: int: approximate number of characters to save from dataset.
    data_keys: Tuple[str]: what keys in dataset to dump from.
  Returns:
    name of temp file with dataset bytes, exact number of characters dumped.
  """
  with tempfile.NamedTemporaryFile(delete=False, prefix="/tmp/ds_chars") as outfp:
    for key in mergeable_ranks:
        line = key + b"\n"
        outfp.write(line)
    for key in special_tokens:
        line = bytes(key, "utf-16") + b"\n"
        outfp.write(line)
  return outfp.name

pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

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
    for i in range(5, num_reserved_special_tokens - 5)
]
special_tokens_dict = {
    token: num_base_tokens + i for i, token in enumerate(special_tokens)
}
fname = _dump_chars_to_textfile(mergeable_ranks,special_tokens_dict)
tiktoken_model = tiktoken.Encoding(
    name=Path(model_path).name,
    pat_str=pat_str,
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens_dict,
)

n_words = tiktoken_model.n_vocab
with tempfile.NamedTemporaryFile(delete=False, prefix="/tmp/sp_tmp") as model_fp:
    pass  # we just want a prefix'd tmp-filename
argstr = " ".join([
      f"--input={fname}",
      f"--vocab_size={n_words}",
      f"--model_prefix={model_fp.name}",
      f"--model_type=bpe",
])
SentencePieceTrainer.Train(argstr)
abs_model_path = 'assets/tokenizer_spm.llama3'
if jax.process_index() == 0:
    # Use an intermediate filename that is renamed to the target name to address
    # create and fill delays.
    copy_rename_path = abs_model_path + ".rntmp"
    tf.io.gfile.copy(model_fp.name + ".model", copy_rename_path, overwrite=True)
    tf.io.gfile.rename(copy_rename_path, abs_model_path, overwrite=True)

def _load_sentencepiece_tokenizer(tokenizer_path: str, add_bos: bool = False, add_eos: bool = True, reverse: bool = False):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  with tf.io.gfile.GFile(tokenizer_path, "rb") as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
  return sp_tokenizer

spm_tokenizer = _load_sentencepiece_tokenizer('assets/tokenizer_spm.llama3', add_bos = False, add_eos=False)
breakpoint()