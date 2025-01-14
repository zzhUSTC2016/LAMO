import pandas as pd
import dill
import openai
import tiktoken
from tqdm import tqdm, trange

encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

def call_api(content):
    """
    call openai api
    :param content: prompt, str
    :return: model response, str
    """
    messages = [{"role": "user", "content": content}]
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0)
    msg = completion.get("choices")[0]["message"]["content"]
    return msg


def get_msg(content):
    """
    get response from openai api
    :param content: prompt, str
    :return: model response, str
    """
    try:
        msg = call_api(content)
    except Exception as e:
        msg = get_msg(content)
    return msg


def simplify_note(note):
    prompt = 'Please summarize specific sections from a patient\'s discharge summary: 1. HISTORY OF PRESENT ILLNESS, 2. PAST MEDICAL HISTORY, 3. ALLERGIES, 4. MEDICATIONS ON ADMISSION 5.DISCHARGE MEDICATIONS. Ignore other details while in hospital and focus only on these sections.\n'\
'output template:\n'\
'HISTORY OF PRESENT ILLNESS:\n'\
'(Language summary as short as possible)\n'\
'PAST MEDICAL HISTORY:\n'\
'(Language summary as short as possible)\n'\
'ALLERGIES:\n'\
'(A series of allergies names, separated by commas, does not require any other information)\n'\
'MEDICATIONS ON ADMISSION:\n'\
'(A series of drug names, separated by commas, remove dosage information. Maybe None.)\n'\
'DISCHARGE MEDICATIONS:\n'\
'(A series of drug names, separated by commas, remove dosage information. Maybe None.)\n'\
'Note:' + note + '\n' + 'Summarize result in five aspects in a concise paragraph without any other words:\n'
    
    msg = get_msg(prompt)

    return msg


def split_string(s, splitted_num):
    split_indices = [i * len(s) // splitted_num for i in range(1, splitted_num)]

    result = []
    start = 0
    for index in split_indices:
        end_0 = min(s.find('.', index), s.find('\n', index))
        end_new = s.find('\n\n', index)
        if abs(end_new - end_0) < 200:
            end = end_new
        else:
            end = end_0
        if end == -1:
            end = len(s)
        result.append(s[start:end + 1])
        start = end + 1

    result.append(s[start:])

    return result

def devide_list(origin_text_list):
    while 1:
        new_text_list = []
        for text in origin_text_list:
            if len(encoding.encode(text)) < 3800:
                new_text_list.append(text)
            else:
                splitted_num = len(encoding.encode(text)) // 3800  + 1
                splitted_result = split_string(text, splitted_num)
                new_text_list += splitted_result
        if new_text_list == origin_text_list:
            break
        else:
            origin_text_list = new_text_list
    return new_text_list

def check_note(note):
    idx1 = note.upper().find('HISTORY OF PRESENT ILLNESS')
    idx2 = note.upper().find('PAST MEDICAL HISTORY')
    idx3 = note.upper().find('ALLERGIES')
    idx4 = note.upper().find('MEDICATIONS ON ADMISSION')
    idx5 = note.upper().find('DISCHARGE MEDICATIONS')
    if idx1 == -1 or idx2 == -1 or idx3 == -1 or idx4 == -1 or idx5 == -1:
        return False
    elif idx1 > idx2 or idx2 > idx3 or idx3 > idx4 or idx4 > idx5:
        return False
    else:
        return True

def generate_note(row):
    # for index, row in result_data.iterrows():
    hadm_id = row['HADM_ID']
    note_text = row.TEXT

    origin_text_list = devide_list([note_text])
    if len(origin_text_list) == 1:
        for i in range(10):
            note = simplify_note(origin_text_list[0])
            if check_note(note):
                break
            else:
                note = simplify_note(origin_text_list[0])
  
        return hadm_id, [note]
    else:
        processed_text = []
        for text_idx, text in enumerate(origin_text_list):
            for i in range(10):
                note = simplify_note(text)
                if check_note(note):
                    break
                else:
                    note = simplify_note(text)
            processed_text.append(note)

        return hadm_id, processed_text
    