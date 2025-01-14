import dill
import fire
import gc
import numpy as np
import os
import sys
import torch 
from vllm import SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

# nohup python each_drug_evaluation.py > /dev/null 2>&1 &

sys.path.append('../')
import utils

def each_lora_evaluation_vllm(run_name, 
                gpu_memory_utilization=0.8, 
                vllm=True,
                mode='val',
                use_history=True,
                use_note=True,
                use_disease=True,
                use_procedure=True,
                base_model="llama-2-7b", 
                file_name='data4LLM.csv'):
    # setup logging
    log_file = f'../output/{run_name}/{mode}.log'
    logger = utils.setup_logger(log_file=log_file, mode='a')

    data_val, med_names = utils.load_data(mode=mode, file_name=file_name)
    output_dir = f'../output/{run_name}/checkpoints/'
    assert os.path.exists(output_dir) == True, "output_dir not exists"
    pred_result_dir = f'../output/{run_name}/pred_result_{mode}'
    lora_path_list = os.listdir(output_dir)
    if '_' in lora_path_list[0]:
        lora_path_dict = {lora_path.split('_')[2]: lora_path for lora_path in lora_path_list if 'checkpoint' in lora_path}
    else:
        lora_path_dict = {lora_path.split('-')[2]: lora_path for lora_path in lora_path_list if 'checkpoint' in lora_path}
    print(lora_path_dict)

    # 获取ground truth
    ground_truth = []
    for i in range(len(data_val)):
        drug_name = data_val.iloc[i]['drug_name'][2:-2].split("', '")
        ground_truth.append(drug_name)
    
    template_name = 'llama-2'

    # setup prompter and evaluator
    prompter = utils.Prompter(template_name=template_name, use_history=use_history, use_note=use_note, use_disease=use_disease, use_procedure=use_procedure)
    evaluator = utils.Evaluator()

    # 先融合所有的lora模型
    for idx, med in enumerate(med_names):
        # 加载vllm模型
        if med not in lora_path_dict.keys():
            print(f'lora path for {idx}-{med} not exists')
            continue
        lora_path = os.path.join(output_dir, lora_path_dict[med])
        model_path = f'../Models/{run_name}/{lora_path_dict[med]}'
        if os.path.exists(lora_path) and not os.path.exists(model_path):
            print(f"------ Merging lora weights from {lora_path} and saving hf model ------")
            utils.merge_lora(base_model=f"../Models/{base_model}",
                            peft_model=lora_path,
                            context_size=8192,
                            save_path=model_path,
                            )
        
    # 获取预测结果
    pred_result = [[] for _ in range(len(data_val))]
    for idx, med in enumerate(med_names):
        # 如果idx不在lora_path_dict中，continue
        if med not in lora_path_dict.keys():
            print(f'lora path for {idx}-{med} not exists')
            continue
        lora_path = os.path.join(output_dir, lora_path_dict[med])
        model_path = f'../Models/{run_name}/{lora_path_dict[med]}'
        # if model_path not exists, continue
        if not os.path.exists(model_path):
            print(f'model path for {idx}-{med} not exists')
            continue
        # 如果已经预测过，加载预测结果
        if os.path.exists(f'{pred_result_dir}/{idx}-{med}.pkl'):
            print(f"------ Loading {idx}-{med} ------")
            with open(f'{pred_result_dir}/{idx}-{med}.pkl', 'rb') as f:
                msg = dill.load(f)
        # 否则，预测
        else:
            print(f"------ Evaluating {idx}-{med}, model_path: {model_path} ------")

            # load model
            # model = LLM(model=model_path,
            #             gpu_memory_utilization=gpu_memory_utilization,
            #             # tensor_parallel_size=3
            #             )
            model = utils.load_LLM(base_model_name=base_model, 
                                    model_path=model_path,
                                    lora_path=lora_path,
                                    vllm=vllm,
                                    gpu_memory_utilization=gpu_memory_utilization)
            sampling_params = SamplingParams(temperature=0, presence_penalty=1)

            # 获取预测结果
            batch = []
            for i, visit in data_val.iterrows():
                if use_history and i > 0 and visit.SUBJECT_ID == data_val.iloc[i-1].SUBJECT_ID:
                    history_info = data_val.iloc[i-1]
                else:
                    history_info = None
                batch.append((history_info, visit, med))
            
            batch_data = list(map(prompter.get_evaluation_prompt, batch))
            msg = model.generate(batch_data, sampling_params)
            if vllm:
                msg = list(map(lambda x: x.outputs[0].text.strip("\'\n"), msg))
            

            # 保存预测结果
            if not os.path.exists(pred_result_dir):
                os.makedirs(pred_result_dir, exist_ok=True)

            with open(f'{pred_result_dir}/{idx}-{med}.pkl', 'wb') as f:
                dill.dump(msg, f)
            

            # 释放显存 
            destroy_model_parallel()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        pred_num = 0
        for i in range(len(msg)):
            if 'Yes' in msg[i] and 'No' not in msg[i]:
                pred_result[i].append(med)
                pred_num += 1


    gt_num = 0
    for i in range(len(ground_truth)):
        if med in ground_truth[i]:
            gt_num += 1

    jaccard_list, f1_list, recall_list, precision_list, \
            ddi_list, drug_num_list, = [], [], [], [], [], []
    # 评估
    for i in range(len(pred_result)):
        recall, precision, f1, jaccard, ddi_rate, drug_num, _ = evaluator.get_metrics(pred_result[i], ground_truth[i])
        jaccard_list.append(jaccard)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        ddi_list.append(ddi_rate)
        drug_num_list.append(drug_num)

    # 保存结果 
    result = f'{idx:3d}-{med:20s}   ' \
                f'jaccard: {np.mean(jaccard_list).round(4)}   ' \
                f'f1: {np.mean(f1_list).round(4)}   ' \
                f'recall: {np.mean(recall_list).round(4)}   ' \
                f'precision: {np.mean(precision_list).round(4)}   ' \
                f'ddi: {np.mean(ddi_list).round(4)}   ' \
                f'drug_num: {np.mean(drug_num_list).round(4)}   ' \
                f'pred_num: {pred_num}   ' 
    logger.info(result)

if __name__ == '__main__':
    fire.Fire(each_lora_evaluation_vllm)