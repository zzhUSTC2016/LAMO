import fire
import json
import numpy as np
import random
import sys
from tqdm import tqdm
sys.path.append('../')
import utils

# fix random seed
np.random.seed(2023)
random.seed(2023)

prompter = utils.Prompter()

def generate_finetune_data(med_name, logger=None, negative_ratio=2, num_val_examples=100):
    if num_val_examples < 0:
        num_val_examples = 10e6

    data_train, med_names, med_weights = utils.load_data(mode='train')
    pos_num_visit = 0
    neg_num_visit = 0
    json_data = []
    for idx, row in data_train.iterrows():
        pos_med_list = eval(row['drug_name'])
        if med_name in pos_med_list:
            pos_num_visit += 1
            prompt = prompter.generate_input(row, drug_candidate=med_name)
            output = 'Yes.'
            json_item = {"input": prompt, "output": output}
            json_data.append(json_item)
        elif neg_num_visit < negative_ratio * pos_num_visit or negative_ratio < 0:
            neg_num_visit += 1
            prompt = prompter.generate_input(row, drug_candidate=med_name)
            output = 'No.'
            json_item = {"input": prompt, "output": output}
            json_data.append(json_item)
    
    data_val, med_names = utils.load_data(mode='val')
    data_len = len(data_val)
    pos_num_visit_val = 0
    neg_num_visit_val = 0
    for idx, row in data_val.iterrows():
        pos_med_list = eval(row['drug_name'])
        if med_name in pos_med_list:
            pos_num_visit_val += 1
            prompt = prompter.generate_input(row, drug_candidate=med_name)
            output = 'Yes.'
            json_item = {"input": prompt, "output": output}
            json_data.append(json_item)
        # elif neg_num_visit_val < negative_ratio * pos_num_visit_val or negative_ratio < 0:
        else:
            neg_num_visit_val += 1
            prompt = prompter.generate_input(row, drug_candidate=med_name)
            output = 'No.'
            json_item = {"input": prompt, "output": output}
            json_data.append(json_item)
        if pos_num_visit_val >= num_val_examples:
            break
        
    file_path = '../data/finetune_data/{}_ratio_{}.json'.format(med_name.replace(' ', '-'), negative_ratio)
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    train_length = pos_num_visit + neg_num_visit
    val_length = pos_num_visit_val + neg_num_visit_val

    if logger is None:
        print(f'Generate finetune data for {med_name} successfully! Train Length: {train_length}={pos_num_visit}+{neg_num_visit}, Val Length: {val_length}={pos_num_visit_val}+{neg_num_visit_val}')
    else:
        logger.info(f'Generate finetune data for {med_name} successfully! Train Length: {train_length}={pos_num_visit}+{neg_num_visit}, Val Length: {val_length}={pos_num_visit_val}+{neg_num_visit_val}')
    return file_path, val_length

if __name__ == "__main__":
    fire.Fire(generate_finetune_data)
