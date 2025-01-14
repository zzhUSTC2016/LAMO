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


def overall_evaluation_vllm(lora_path, gpu_memory_utilization=0.96, data_len=100, logger=None, remove_model=True, base_model="llama-2-7b"):
    if 'checkpoint' not in lora_path:
        checkpoints = [folder for folder in os.listdir(
            lora_path) if 'checkpoint' in folder]
        assert len(checkpoints) > 0, "No checkpoint found"
        lora_path = os.path.join(lora_path, checkpoints[0])

    model_path = lora_path.replace('output', 'Models')
    model_name = model_path.split('/')[-1]

    assert os.path.exists(lora_path) or os.path.exists(
        model_path), "No model found"

    if os.path.exists(lora_path) and not os.path.exists(model_path):
        print("------ Overall evaluation: Merging lora weights and saving hf model ------")
        utils.merge_lora(base_model=f"../Models/{base_model}",
                         peft_model=lora_path,
                         context_size=8192,
                         save_path=model_path,
                         )
        # print("------ Done merging lora weights and saving hf model ------\n")
    
    if 'llama' in base_model:
        template_name = 'llama-2'
    data, med_names = utils.load_data(mode='test', file_name='data4LLM_CONCISE_NOTE.csv')

    # record the number of each medicine predicted by the model
    med_dict_pred = {item: 0 for item in med_names}
    med_dict_gt = {item: 0 for item in med_names}

    # setup logging
    # log_file = f'../log/{model_name}.log'
    # utils.set_logger(log_file)

    # load model
    model = LLM(model=model_path,
                gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(temperature=0, presence_penalty=1)

    # setup prompter
    prompter = utils.Prompter(template_name=template_name)

    # setup evaluator
    evaluator = utils.Evaluator()

    if data_len > 0:
        data = data[:data_len]

    # 生成组合
    # patient_med_combinations = list(product(data.iterrows(), med_names))
    patient_med_combinations = []
    for idx, row in data.iterrows():
        if idx > 0 and row['SUBJECT_ID'] == data.iloc[idx - 1]['SUBJECT_ID']:
            history_info = data.iloc[idx - 1]
        else:
            history_info = None
        for med_name in med_names:
            patient_med_combinations.append((history_info, row, med_name))

    # remove idx in patient_med_combinations
    # patient_med_combinations = list(
    #     map(lambda x: (x[0][1], x[1]), patient_med_combinations))

    jaccard_list, f1_list, recall_list, precision_list, \
        ddi_list, drug_num_list, drug_num_gt_list, refuse_list = [], [], [], [], [], [], [], []
    # for each visit
    prompts = list(map(prompter.get_evaluation_prompt,
                   patient_med_combinations))
    # print(prompts)
    msg_raw = model.generate(prompts, sampling_params)

    msg = list(map(lambda x: x.outputs[0].text.strip("\'\n"), msg_raw))
    for idx, visit in data.iterrows():
        pred_list = []
        refuse_num = 0
        gt_list = visit.drug_name[2:-2].split("', '")
        visit_data = patient_med_combinations[idx *
                                              len(med_names):(idx + 1) * len(med_names)]
        msg_list = msg[idx * len(med_names):(idx + 1) * len(med_names)]

        for i in range(len(msg_list)):
            if 'Yes' in msg_list[i] and 'No' not in msg_list[i]:
                pred_list.append(med_names[i])
                med_dict_pred[med_names[i]] += 1
            elif 'No' in msg_list[i] and 'Yes' not in msg_list[i]:
                pass
            else:
                refuse_num += 1

        for med in gt_list:
            med_dict_gt[med] += 1

        recall, precision, f1, jaccard, ddi_rate, drug_num, drug_num_gt = evaluator.get_metrics(
            pred_list, gt_list)
        jaccard_list.append(jaccard)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        ddi_list.append(ddi_rate)
        drug_num_list.append(drug_num)
        drug_num_gt_list.append(drug_num_gt)
        refuse_list.append(refuse_num/len(med_names))

        performance = textwrap.dedent(f""" 
                    jaccard: {np.average(jaccard_list):.4f}, 
                    recall: {np.average(recall_list):.4f}, 
                    precision: {np.average(precision_list):.4f}, 
                    f1: {np.average(f1_list):.4f}, 
                    ddi_rate: {np.average(ddi_list):.4f}, 
                    drug_num: {np.average(drug_num_list):.2f}/{len(med_names)}, 
                    drug_num_gt: {np.average(drug_num_gt_list):.2f}/{len(med_names)},
                    refuse_rate: {np.average(refuse_list):.4f}
                    """).replace("\n", "")
        if idx % 10 == 0:
            if logger is None:
                print(idx, performance)
            else:
                logger.info(performance)

    if logger is None:
        print(performance)
    else:
        logger.info(performance)

    # plot and log the number of each medicine predicted by the model
    med_dict_gt = {k: v for k, v in sorted(
        med_dict_gt.items(), key=lambda item: item[1], reverse=True)}
    # sort med_dict_pred with the same key order as med_dict_gt
    med_dict_pred = {k: med_dict_pred[k] for k in med_dict_gt.keys()}
    # plt.figure(figsize=(20, 10))
    x = np.arange(len(med_dict_gt))
    plt.plot(x, list(med_dict_gt.values()), label='gt', color='blue')
    plt.plot(x, list(med_dict_pred.values()),
                label='pred', color='red')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../log/{model_name}.png')
    plt.show()

    dill.dump(med_dict_gt, open(f'../log/{model_name}_med_dict_gt.pkl', 'wb'))
    dill.dump(med_dict_pred, open(f'../log/{model_name}_med_dict_pred.pkl', 'wb'))

    destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # delete model_path
    model_path = '/'.join(model_path.split('/')[:3])
    # 如果不以llama开头，就删除
    if 'llama' not in model_path and remove_model:
        os.system(f"rm -rf {model_path}")


if __name__ == '__main__':
    fire.Fire(overall_evaluation_vllm)


# run this script with a specified cuda:
# CUDA_VISIBLE_DEVICES=5 python evaluation_vllm.py --model_name /ratio_5_3e-5/checkpoint-2016  --gpu_memory_utilization 0.4 --data_len 100
