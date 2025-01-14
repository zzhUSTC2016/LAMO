import os
import sys
from typing import List
import fire
import numpy as np
import logging
import torch
import transformers
from datasets import load_dataset
sys.path.append('../')

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils import Prompter
import utils

from transformers import EarlyStoppingCallback, TrainerCallback, \
    TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# CUDA_VISIBLE_DEVICES=3,4,5,6 nohup torchrun --nproc_per_node=4 --master_port=29501 finetune.py --learning_rate 5e-5 --num_epochs 1 --train_on_inputs 0 --data_path ../data/finetune_data_ratio_10.json --output_dir ../output/ratio_10_ips/  > ratio10_ips.out 2> ratio10_ips.err &  

# torchrun --nproc_per_node=4 --master_port=29501 finetune_fix_drug.py --learning_rate 5e-5 --num_epochs 5 --val_set_size 704 --batch_size 64 --train_on_inputs 0 --resume_from_checkpoint 0 --data_path ../data/finetune_data_ratio_10_fix_200.json --output_dir ../output/ratio_10_fix_200/  > ratio_10_fix_200.out 2> ratio_10_fix_200.err

# CUDA_VISIBLE_DEVICES=2,3 nohup torchrun --nproc_per_node=2 --master_port=29502 finetune.py --data_path ../data/finetune_data_test.json --output_dir ../output/tmp/ --eval_epochs 4 --num_epochs 20   --early_stopping_patience 3 --val_set_size 716 > test.out 2> test.err &

    

def train(
    # model/data params
    base_model: str = "llama-2-7b",  # the only required argument
    data_path: str = "../data/finetune_data_ratio_7.json",
    output_dir: str = "../output/ratio_7_3e-5_input/",
    load_in_8bit: bool = False,
    eval_epochs: int = 32,   
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 2,
    num_epochs: int = 1,   
    learning_rate: float = 3e-5,
    cutoff_len: int = 5120,
    med_num: int = 151,
    val_set_size: int = 704,
    random_jaccard: float = 0.0,
    early_stopping_patience: int = 10,  
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    # llm hyperparams
    train_on_inputs: int = 0,  # if False, masks out inputs in loss
    resume_from_checkpoint: int = 0,  # either training checkpoint or final adapter
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    use_flash_attn: bool = True, # whether to use Flash Attention
    logging_dir: str = None,
    ):

    base_model = f"../Models/{base_model}"

    log_file = os.path.join(output_dir, "train.log")
    logger = utils.setup_logger(log_file, mode='a')

    train_on_inputs = True if train_on_inputs else False
    resume_from_checkpoint = True if resume_from_checkpoint else False

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training LLaMA-2 model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"eval_epochs: {eval_epochs}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"early_stopping_patience: {early_stopping_patience}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"use_flash_attn: {use_flash_attn}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='../Models/llama-2-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # prompt_template_name = base_model.split("/")[-1]
    prompt_template_name = 'llama-2'
    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        use_flash_attention_2=use_flash_attn,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=False,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["history"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["history"],
                data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could not speed up
        return tokenized_full_prompt

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=False
        )
        train_data = (
            train_val["train"].map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].map(generate_and_tokenize_prompt)
        val_data = None
    # shuffle the training and validation data
    train_data = train_data.shuffle(seed=42)
    if val_data is not None:
        val_data = val_data.shuffle(seed=42)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_folder)
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control

    # class TrainBeginCallback(TrainerCallback):
    #     def on_train_begin(
    #         self,
    #         args: TrainingArguments,
    #         state: TrainerState,
    #         control: TrainerControl,
    #         **kwargs,
    #     ):
    #         state.best_metric = None
    #         print(f'reset state.best_metric to: {state.best_metric}')
    #         print(state)
    #         return control

    class MyEarlyStoppingCallback(EarlyStoppingCallback):
        def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold=0.01):
            super().__init__(early_stopping_patience=early_stopping_patience, 
                             early_stopping_threshold=early_stopping_threshold)

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics.get(metric_to_check)

            if metric_value is None:
                logger.warning(
                    f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                    " is disabled"
                )
                return

            self.check_metric_value(args, state, control, metric_value)
            if state.global_step > 320 and self.early_stopping_patience_counter >= self.early_stopping_patience and state.best_metric > random_jaccard:
                control.should_training_stop = True
    
    # class EvaluateFirstStepCallback(TrainerCallback):
    #     def on_step_end(self, args, state, control, **kwargs):
    #         if state.global_step == 1:
    #             control.should_evaluate = True
    #             control.should_save = True


    def metrics_per_visit(preds, labels):
        recall = (preds * labels).sum() / labels.sum()
        precision = (preds * labels).sum() / preds.sum() if preds.sum() > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0
        jaccard = (preds * labels).sum() / ((preds + labels) > 0).sum()
        num_drugs = preds.sum()
        num_drugs_gt = labels.sum()
        num_visits = preds.shape[0]
        return recall, precision, f1, jaccard, num_drugs, num_drugs_gt, num_visits

    # Metric
    def compute_metrics(eval_preds):
        (preds, labels), _ = eval_preds
        recall, precision, f1, jaccard, num_drugs, num_drugs_gt, num_visits = metrics_per_visit(preds, labels)
        return {
            "num_drugs": num_drugs,
            "num_drugs_gt": num_drugs_gt,
            "num_visits": num_visits,
            "jaccard": round(jaccard, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        This function is used to preprocess logits for the compute_metrics function.
        Output: 
            logits: [batch_size], binary logits, 0 for 3782 (No), 1 for 8241 (Yes), \
                indicating whether the drug is predicted or not
            labels: [batch_size], binary labels, \
                indicating whether the drug is prescribed or not
        """
        # labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))

        # get the last index of 8241 or 3782 in each row
        condition = torch.bitwise_or(labels == 8241, labels == 3782)
        indices = torch.nonzero(condition)
        result = []
        # print(f"{int(os.environ.get('LOCAL_RANK') or 0)}, {logits.shape}, {labels.shape}, {condition.shape}, {indices.shape}")
        # if len(indices) < 24:
        #     print(f"{int(os.environ.get('LOCAL_RANK') or 0)}, {indices}, {labels[5]}")
        #     os.exit(0)

        for i in range(labels.size(0)):
            row_indices = indices[indices[:, 0] == i][:, 1]
            if len(row_indices) > 0:
                last_index = row_indices[-1]
                if labels[i, last_index] == 8241 or labels[i, last_index] == 3782:
                    result.append([i, last_index])

        labels_index = torch.tensor(result)

        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
        labels_index[: , 1] = labels_index[: , 1] - 1
        logits = logits.softmax(dim=-1)
        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3782, 8241]], dim = -1)
        logits = torch.argmax(logits, dim=-1)
        return logits, gold

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy="steps",
            # warmup_ratio=0.1,
            warmup_steps=10,
            logging_steps=eval_epochs,
            save_steps=eval_epochs,
            eval_steps=eval_epochs if val_set_size > 0 else None,
            save_total_limit=1,
            metric_for_best_model='jaccard',
            # fp16=True,
            bf16=True,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            optim="adamw_torch",
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False, # if ddp else None,
            group_by_length=group_by_length,
            report_to="tensorboard",
            log_level="info",
            ignore_data_skip=True,
            logging_dir=logging_dir,
            lr_scheduler_type='inverse_sqrt',
            log_on_each_node=False,
            logging_first_step=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[SavePeftModelCallback, 
                   MyEarlyStoppingCallback(
                        early_stopping_patience=early_stopping_patience,
                        early_stopping_threshold=0.0),
                #    EvaluateFirstStepCallback()
                   ],
    )
    model.config.use_cache = False

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logger.info(f"Training on {data_path} complete, best metric: {trainer.state.best_metric}, saved in: {trainer.state.best_model_checkpoint}")
    # model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)



