import dill
import fire
from functools import partial
import matplotlib.pyplot as plt
import gc
import importlib
import logging
import numpy as np
import sys
import subprocess
import textwrap
from tqdm import tqdm
from itertools import product
from vllm import LLM, SamplingParams
import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import os
sys.path.append('../')
import utils


def each_drug_evaluation_vllm(lora_path, 
                              test_med_list, 
                              gpu_memory_utilization=0.96, 
                              data_len=-1, 
                              logger=None, 
                              remove_model=True,
                              vllm=True,
                              use_note=True,
                              start_idx=0,
                              end_idx=-1,
                              base_model="llama-2-7b",
                              file_name='data4LLM.csv'):
    """
    Evaluate the performance of the model on the drug prediction task.
    :param lora_path: checkpoint path or folder path containing checkpoint, e.g., ../output/ratio_5_3e-5/checkpoint-2016 or ../output/ratio_5_3e-5
    :param test_med_list: the list of drugs to be evaluated
    :param gpu_memory_utilization: the gpu memory utilization
    :param data_len: the length of data to be evaluated
    :return: None
    """
    # 寻找出lora_path字符串output/后，/checkpoint前的字符串：
    start = lora_path.find('output/')
    end = lora_path.find('/checkpoint')
    lora_name = lora_path[start + len('output/'):end]

    # setup logging
    if logger is None:
        if start_idx == 0 and end_idx == -1:
            log_file = f'../output/{lora_name}/each_drug_evaluation.log'
            logger = utils.setup_logger(log_file=log_file)
        else:
            log_file = f'../output/{lora_name}/each_drug_evaluation_{start_idx}_{end_idx}.log'
            logger = utils.setup_logger(log_file=log_file)

    if 'checkpoint' not in lora_path:
        checkpoints = [folder for folder in os.listdir(
            lora_path) if 'checkpoint' in folder]
        assert len(checkpoints) > 0, "No checkpoint found"
        lora_path = os.path.join(lora_path, checkpoints[0])

    model_path = lora_path.replace('output', 'Models')

    assert os.path.exists(lora_path) or os.path.exists(
        model_path), "No model found"

    if os.path.exists(lora_path) and not os.path.exists(model_path):
        print(
            "------ Single drug evaluation: Merging lora weights and saving hf model ------")
        utils.merge_lora(base_model=f"../Models/{base_model}",
                         peft_model=lora_path,
                         context_size=8192,
                         save_path=model_path,
                         )
        # print("------ Done merging lora weights and saving hf model ------\n")
    
    if 'llama' in base_model:
        template_name = 'llama-2'
    data, med_names = utils.load_data(mode='test', file_name=file_name)

    if test_med_list is None:
        test_med_list = med_names
    
    
    # load model
    # model = LLM(model=model_path,
    #             gpu_memory_utilization=gpu_memory_utilization,
    #             )
    model = utils.load_LLM(base_model_name=base_model, 
                                    model_path=model_path,
                                    lora_path=lora_path,
                                    vllm=vllm,
                                    gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(temperature=0, presence_penalty=1)

    # setup prompter
    prompter = utils.Prompter(template_name=template_name, use_note=use_note)

    # setup evaluator
    evaluator = utils.Evaluator()

    # for each visit
    if data_len > 0:
        data = data[:data_len+1]

    jaccard_list = []

    for test_med in test_med_list:
        pred_visit, gt_visit = [], []
        batch = []

        med_idx = med_names.index(test_med)
        if start_idx > med_idx:
            continue
        if end_idx != -1 and med_idx >= end_idx:
            break
        for idx, visit in data.iterrows():
            if idx > 0 and visit.SUBJECT_ID == data.iloc[idx-1].SUBJECT_ID:
                history_info = data.iloc[idx-1]
            else:
                history_info = None
            gt_list = visit.drug_name[2:-2].split("', '")
            if test_med in gt_list:
                gt_visit.append(visit.HADM_ID)

            batch.append((history_info, visit, test_med))

        batch_data = list(map(prompter.get_evaluation_prompt, batch))
        msg = model.generate(batch_data, sampling_params)
        if vllm:
            msg = list(map(lambda x: x.outputs[0].text.strip("\'\n"), msg))
        # print(msg)

        for i in range(len(msg)):
            if 'Yes' in msg[i] or 'Ye' in msg[i] and 'No' not in msg[i]:
                pred_visit.append(batch[i][1].HADM_ID)

        recall, precision, f1, jaccard, _, _, _ = evaluator.get_metrics(
            pred_visit, gt_visit)
        drug_num = len(pred_visit)
        drug_num_gt = len(gt_visit)
        jaccard_list.append(jaccard)
        if logger is None:
            print(f'{med_names.index(test_med)}-{test_med:24}, recall:{recall:.4f}  precision:{precision:.4f}  f1:{f1:.4f}  jaccard:{jaccard:.4f}  drug_pred:{drug_num}/{len(data)}, drug_gt:{drug_num_gt}')
        else:
            logger.info(
                f'{med_names.index(test_med)}-{test_med:24}, recall:{recall:.4f}  precision:{precision:.4f}  f1:{f1:.4f}  jaccard:{jaccard:.4f}  drug_pred:{drug_num}/{len(data)}, drug_gt:{drug_num_gt}')

    destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # print(f'jaccard: {np.mean(jaccard_list).round(4)}')
    # print(f'len(jaccard_list): {len(jaccard_list)}')
    # print(f'jaccard_list: {jaccard_list}')

    # delete model_path
    # 保留两层目录
    model_path = '/'.join(model_path.split('/')[:3])
    # 如果不以llama开头，就删除
    if 'llama' not in model_path and remove_model:
        os.system(f"rm -rf {model_path}")

    return f1


if __name__ == '__main__':
    fire.Fire(each_drug_evaluation_vllm)


# run this script with a specified cuda:
# CUDA_VISIBLE_DEVICES=5 python drug_evaluation_vllm.py --model_name /ratio_5_3e-5/checkpoint-2016  --gpu_memory_utilization 0.4 --data_len 100
