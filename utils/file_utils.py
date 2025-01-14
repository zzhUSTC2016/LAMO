import json
import logging
import os
import pandas as pd


def setup_logger(log_file, mode='w'):
    # create log folder if not exists
    log_folder = os.path.dirname(log_file)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def load_data(mode='train', file_name='data4LLM.csv'):
    '''
    Load EHR data and drug names
    '''
    # load EHR data
    data_path = '../data/mimic-iii/'

    data4LLM_file = os.path.join(data_path, file_name)
    data = pd.read_csv(data4LLM_file)

    # split data into train, val, test
    split_point = int(len(data) * 2 / 3)
    val_len = int(len(data[split_point:]) / 2)
    data_train = data[:split_point]
    data_val = data[split_point:split_point+val_len]
    data_val.reset_index(inplace=True, drop=True)
    data_test = data[split_point+val_len:]
    data_test.reset_index(inplace=True, drop=True)

    # get all drug names
    med_count = dict()
    for med in data['drug_name']:
        med_list = eval(med)
        for med in med_list:
            if med not in med_count:
                med_count[med] = 1
            else:
                med_count[med] += 1
    # sort the med_count by value
    med_count = {k: v for k, v in sorted(
        med_count.items(), key=lambda item: item[1], reverse=True)}

    med_names = list(med_count.keys())
    med_weights = [med_count[med] for med in med_names]
    # print(f'Total number of drugs: {len(med_names)}')   # 151

    # return data and drug names
    if mode == 'train':
        # print(f'Train data size: {len(data_train)}\n')    # 9960
        return data_train, med_names, med_weights
    elif mode == 'val':
        # print(f'Val data size: {len(data_val)}\n')     # 2490
        return data_val, med_names
    elif mode == 'test':
        print(f'Test data size: {len(data_test)}\n')
        return data_test, med_names
    else:
        raise ValueError('Wrong mode!')



def rename_checkpoint(output_dir, med, idx, jaccard, remove_checkpoint=True):
    # 把output_dir下带有'checkpoint'的文件夹重命名为checkpoint-现有药物的名字，并移动到finetune_results文件夹下
    os.makedirs(os.path.join(output_dir, 'finetune_results'), exist_ok=True)

    checkpoint_list = [folder for folder in os.listdir(output_dir) if 'checkpoint' in folder]
    assert len(checkpoint_list) > 0, "No checkpoint found"
    # assert len(checkpoint_list) == 1, "More than one checkpoint found"
    if len(checkpoint_list) > 1:
        print(f'More than one checkpoint found: {checkpoint_list}, use the first one: {checkpoint_list[0]}')

    checkpoint_name = checkpoint_list[0]

    # copy and rename the checkpoint folder
    new_folder_name = checkpoint_name.replace('checkpoint', f'{idx}_checkpoint_{med}_{jaccard:.4f}')
    os.system(f'cp -r {os.path.join(output_dir, checkpoint_name)} {os.path.join(output_dir, "finetune_results")}')
    if os.path.exists(os.path.join(output_dir, 'finetune_results', new_folder_name)):
        os.system(f'rm -r {os.path.join(output_dir, "finetune_results", new_folder_name)}')
    os.rename(os.path.join(output_dir, 'finetune_results', checkpoint_name), os.path.join(output_dir, 'finetune_results', new_folder_name))

    # load and modify the trainer_state.json file to reset the best_metric, global_step and epoch, avoiding the wrong early stopping
    trainer_state_path = os.path.join(output_dir, checkpoint_name, 'trainer_state.json')
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    trainer_state['best_metric'] = 0.0
    trainer_state['global_step'] = 0
    trainer_state['epoch'] = 0
    trainer_state['log_history'] = []
    with open(trainer_state_path, 'w') as f:
        json.dump(trainer_state, f)

    if remove_checkpoint:
        # remove the checkpoint folder
        for checkpoint_name in checkpoint_list:
            os.system(f'rm -r {os.path.join(output_dir, checkpoint_name)}')
    else:
        # delete the scheduler.pt file to reset the lr_scheduler
        os.system(f'rm {os.path.join(output_dir, checkpoint_name, "scheduler.pt")}')
    

def get_master_port():
    # get the master port for torch.distributed.launch
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    return s.getsockname()[1]
