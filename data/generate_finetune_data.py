import fire
import json
import numpy as np
import random
import os
import sys
from tqdm import tqdm
sys.path.append('../')
import utils

# fix random seed
np.random.seed(2023)
random.seed(2023)


def add_candidate_drug(prompter, data, candidate_drug, json_data, 
                       negative_ratio=-1,
                       max_pos_num_visit=10e6, max_visit=10e6):
    pos_num_visit = 0
    neg_num_visit = 0
    for idx, row in data.iterrows():
        pos_med_list = eval(row['drug_name'])
        history = None
        if idx > 0 and data.iloc[idx-1]['SUBJECT_ID'] == row['SUBJECT_ID']:
            history = prompter.generate_history(data.iloc[idx-1])
        prompt = prompter.generate_input(row, drug_candidate=candidate_drug)
        if candidate_drug in pos_med_list:
            pos_num_visit += 1
            output = 'Yes.'
        elif negative_ratio < 0 or (negative_ratio > 0 and neg_num_visit < negative_ratio * pos_num_visit):
            neg_num_visit += 1
            output = 'No.'
        else:
            continue
        if history is not None:
            json_item = {"history": history, "input": prompt, "output": output}
        else:
            json_item = {"history": None, "input": prompt, "output": output}
        json_data.append(json_item)
        if pos_num_visit >= max_pos_num_visit:
            break
        if pos_num_visit + neg_num_visit >= max_visit:
            break
    return json_data, pos_num_visit, neg_num_visit


def add_patients(prompter, data, med_names, num_visits, json_data):
    pos_num_visit = 0
    neg_num_visit = 0
    for idx, row in data.iloc[:num_visits].iterrows():
        pos_med_list = eval(row['drug_name'])
        history = None
        if idx > 0 and data.iloc[idx-1]['SUBJECT_ID'] == row['SUBJECT_ID']:
            history = prompter.generate_history(data.iloc[idx-1])
        for med in med_names:
            prompt = prompter.generate_input(row, drug_candidate=med)
            if med in pos_med_list:
                pos_num_visit += 1
                output = 'Yes.'
            else:
                neg_num_visit += 1
                output = 'No.'
            if history is not None:
                json_item = {"history": history,
                             "input": prompt, "output": output}
            else:
                json_item = {"history": None,
                             "input": prompt, "output": output}
            json_data.append(json_item)
    return json_data, pos_num_visit, neg_num_visit
            


def generate_finetune_data(candidate_drugs, logger=None, negative_ratio=-1, 
                           num_val_examples=100, num_val_visits=20, file_name='data4LLM.csv',
                           use_history=True, use_note=True):
    """
    Generate finetune data for each drug
    :param candidate_drugs: list of candidate drugs
    :param logger: logger
    :param negative_ratio: negative ratio
    :param num_val_examples: 
    """
    prompter = utils.Prompter(use_history=use_history, use_note=use_note)

    # if num_val_examples < 0, use all val data
    if num_val_examples < 0:
        num_val_examples = 10e6
    # if candidate_drugs is str, make it a list
    if isinstance(candidate_drugs, str):
        candidate_drugs = [candidate_drugs]

    data_train, med_names, med_weights = utils.load_data(mode='train', file_name=file_name)
    data_val, med_names = utils.load_data(mode='val', file_name=file_name)
    if logger is None:
        print(f'Loaded data from {file_name}')
    else:    
        logger.info(f'Loaded data from {file_name}')
    json_data = []

    if candidate_drugs is None:
        candidate_drugs = med_names

    pos_num_visit, neg_num_visit = 0, 0
    pos_num_visit_val, neg_num_visit_val = 0, 0
    # train set
    for candidate_drug in tqdm(candidate_drugs, desc='Generating finetune data', total=len(candidate_drugs)):
        json_data, pos_num_visit_, neg_num_visit_ = add_candidate_drug(
            prompter, data_train, candidate_drug, json_data, negative_ratio=negative_ratio)
        pos_num_visit += pos_num_visit_
        neg_num_visit += neg_num_visit_            
    
    # val set
    if len(candidate_drugs) == 1:
        json_data, pos_num_visit_val, neg_num_visit_val = add_candidate_drug(
            prompter, data_val, candidate_drugs[0], json_data, max_pos_num_visit=num_val_examples)
    elif len(candidate_drugs) == len(med_names):
        json_data, pos_num_visit_val, neg_num_visit_val = add_patients(
            prompter, data_val, med_names, num_val_visits, json_data)
    else:
        for drug in candidate_drugs:
            json_data, pos_num_visit_val_, neg_num_visit_val_ = add_candidate_drug(
                prompter, data_val, drug, json_data, max_visit=num_val_examples)
            pos_num_visit_val += pos_num_visit_val_
            neg_num_visit_val += neg_num_visit_val_

    if len(candidate_drugs) == len(med_names):
        file_name = f'all_ratio_{negative_ratio}'
    elif len(candidate_drugs) == 1:
        med_idx = med_names.index(candidate_drugs[0])
        file_name = f"{med_idx}-{candidate_drugs[0].replace(' ', '-')}"
    elif len(candidate_drugs) > 4:
        med_idx = med_names.index(candidate_drugs[0])
        file_name = f"{len(candidate_drugs)}_start_{med_idx}"
    else:
        file_name = '-'.join(candidate_drugs).replace(' ', '-')
    # create folder finetune_data
    os.makedirs('../data/finetune_data/', exist_ok=True)
    file_path = '../data/finetune_data/{}.json'.format(file_name)
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    train_length = pos_num_visit + neg_num_visit
    val_length = pos_num_visit_val + neg_num_visit_val

    if logger is None:
        print(f'Generate finetune data for {file_name} successfully! Train Length: {train_length}={pos_num_visit}+{neg_num_visit}, Val Length: {val_length}={pos_num_visit_val}+{neg_num_visit_val}')
    else:
        logger.info(f'Generate finetune data for {file_name} successfully! Train Length: {train_length}={pos_num_visit}+{neg_num_visit}, Val Length: {val_length}={pos_num_visit_val}+{neg_num_visit_val}')
    return file_path, pos_num_visit_val, neg_num_visit_val

if __name__ == "__main__":
    fire.Fire(generate_finetune_data)
