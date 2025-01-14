# used in evaluation without VLLM

import gc
import torch


def get_msg(inputs, model, tokenizer, device):
    input_ids = tokenizer(inputs,  padding=True,
                        return_tensors="pt", add_special_tokens=False).input_ids.to(device) #将输入的文本转换为token

    generate_input = {
        "input_ids": input_ids, #输入的token
        "max_new_tokens": 20,  #最大生成的token数量
        "do_sample": False,      #是否采样
        "repetition_penalty": 1,               #重复惩罚
        "eos_token_id": tokenizer.eos_token_id,  #结束token
        "bos_token_id": tokenizer.bos_token_id,  #开始token
        "pad_token_id": tokenizer.pad_token_id   #pad token
    }
    generate_ids = model.generate(**generate_input) #生成token
    outputs = tokenizer.batch_decode(generate_ids) #将token转换为文本
    for idx, output in enumerate(outputs):
        outputs[idx] = outputs[idx].strip('</s> ')
        inputs[idx] = inputs[idx].strip('</s> ')
        input = inputs[idx]
        output = outputs[idx]
        # print(output, input)
        if output.startswith(input):
            outputs[idx] = output[len(input):]
        outputs[idx] = outputs[idx].strip(' \n\'')
    gc.collect()
    torch.cuda.empty_cache()
    
    return outputs
