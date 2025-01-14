# Written by Yukang Chen
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

import fire
import gc
import os
import torch
import argparse
import transformers
from peft import PeftModel
from typing import Dict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data/pretrained-models/llama-7b-hf")
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--save_path', type=str, default=None, help='')
    parser.add_argument('--cache_dir', type=str, default=None, help='./cache_dir')
    args = parser.parse_args()
    return args

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def merge_lora(base_model, peft_model, context_size, save_path, cache_dir=None):

    # print("base model", base_model)
    # print("peft model", peft_model)

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        model_max_length=context_size,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    trainable_params = os.path.join(peft_model, "trainable_params.bin")
    if os.path.isfile(trainable_params):
        model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
    model = PeftModel.from_pretrained(
        model,
        peft_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    fire.Fire(merge_lora)


# !CUDA_VISIBLE_DEVICES={cuda_device} \
#     python ../utils/merge_lora_weights_and_save_hf_model.py \
#     --base_model "../Models/llama-2-7b" \
#     --peft_model ../output/{model_name} \
#     --context_size 8192 \
#     --save_path ../Models/{model_name}