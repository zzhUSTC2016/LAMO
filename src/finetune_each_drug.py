import fire
import os
import sys
import logging
import torch

os.environ['MKL_THREADING_LAYER'] = 'GNU'
sys.path.append('../')
import utils
from data.generate_finetune_data import generate_finetune_data
from evaluation._1_overall_evaluation_vllm import overall_evaluation_vllm
from evaluation._2_each_drug_evaluation_vllm import each_drug_evaluation_vllm

def finetune_each_drug(model_name='each_drug', idx_start=None, idx_end=None, negative_ratio=-1, selected_drugs=None, base_model='llama-2-7b'):
    if idx_start is not None or idx_end is not None:
        assert idx_start is not None and idx_end is not None, "idx_start and idx_end must be both specified"
        assert idx_start < idx_end, "idx_start must be smaller than idx_end"
        assert idx_start >= 0 and idx_end <= 151, "idx_start and idx_end must be in [0, 151)"
        model_name = f'each_drug_{idx_start}_{idx_end}'

    output_dir = f'../output/{model_name}'
    # file_name='data4LLM.csv'
    file_name='data4LLM_CONCISE_NOTE.csv'

    # setup logging
    log_file = os.path.join(output_dir, "train.log")
    logger = utils.setup_logger(log_file, mode='a')

    _, med_names, _ = utils.load_data(file_name=file_name)
    med_list = med_names
    for idx, med in enumerate(med_list):
        if selected_drugs is not None and idx not in selected_drugs:
            continue
        elif selected_drugs is None:
            if idx_start is not None and idx < idx_start:
                continue
            if idx_end is not None and idx >= idx_end:
                break
        
        # generate finetune data
        data_path, pos_num_visit_val, neg_num_visit_val = generate_finetune_data(med, logger, 
                                                         negative_ratio=negative_ratio,
                                                         num_val_examples=500, 
                                                         file_name=file_name,
                                                         use_history=False,
                                                         use_note=True)    
        val_set_size = pos_num_visit_val + neg_num_visit_val
        random_jaccard = pos_num_visit_val / val_set_size
        logging_dir = os.path.join(output_dir, 'runs', f'{idx}_{med.replace(" ", "-")}')
        resume_from_checkpoint = 0
        logger.info(f'Training on drug: {idx}-{med}, random metric: {random_jaccard:.4f}')
        os.system(f"torchrun --nproc_per_node={torch.cuda.device_count()} --master_port={utils.get_master_port()} finetune_fix_drug.py --base_model {base_model} --learning_rate 5e-4 --num_epochs 20 --early_stopping_patience 10 --eval_epochs 32 --val_set_size {val_set_size} --random_jaccard {random_jaccard} --batch_size 64 --micro_batch_size 1 --train_on_inputs 0 --resume_from_checkpoint {resume_from_checkpoint} --data_path {data_path} --output_dir {output_dir} --logging_dir {logging_dir}  > {model_name}.out 2> {model_name}.err")

        jaccard = each_drug_evaluation_vllm(output_dir, med_list[idx:idx+1], gpu_memory_utilization=0.8, data_len=-1, logger=logger, remove_model=True, file_name=file_name)
        # overall_evaluation_vllm(output_dir, gpu_memory_utilization=0.8, data_len=20, logger=logger, remove_model=True)
        utils.rename_checkpoint(output_dir, med, idx, jaccard)
        logger.info(f'----------Finished training on drug: {idx}-{med}-----------\n\n')



if __name__ == '__main__':
    fire.Fire(finetune_each_drug)
