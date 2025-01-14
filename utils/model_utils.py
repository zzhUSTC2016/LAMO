from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM
from peft import PeftModel
import torch
import gc
from tqdm import tqdm


def load_model(model_name):
    model_path = f'../Models/{model_name}'
    if model_name == 'medalpaca-13b':
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map = 'auto')
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'auto')
    else:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_path, padding_side='left', device_map = 'auto')
        model = LlamaForCausalLM.from_pretrained(model_path, device_map = 'auto')
    # 为了防止生成的文本出现[PAD]，这里将[PAD]重置为[EOS]
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class MyLLM():
    def __init__(self, base_model_name, model_path=None, lora_path=None):
        base_model, tokenizer = load_model(base_model_name)
        if lora_path is None:
            self.model = base_model
            self.tokenizer = tokenizer
            print(f'Load base model: {base_model_name}')
        else:
            print(f'Load base model: {base_model_name}, lora model: {lora_path}')
            model = PeftModel.from_pretrained(
                                                base_model,
                                                lora_path,
                                                device_map="auto",
                                                torch_dtype=torch.float16,
                                            )
            self.model = model.merge_and_unload()
            self.tokenizer = tokenizer

    def get_msg(self, inputs, model, tokenizer):
        tokenizer_outputs = tokenizer(inputs,  padding=True,
                            return_tensors="pt", add_special_tokens=False)
        input_ids = tokenizer_outputs.input_ids.to('cuda') #将输入的文本转换为token
        attention_mask = tokenizer_outputs.attention_mask.to('cuda')
        generate_input = {
            "input_ids": input_ids, #输入的token
            "attention_mask": attention_mask,
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
    
        return outputs, generate_ids

    def generate(self, batch_data, *args, **kwargs):
        batch_size = 10
        msg = []
        for i in tqdm(range(0, len(batch_data), batch_size)):
            batch_data_ = batch_data[i:i+batch_size]
            msg_, _ = self.get_msg(batch_data_, self.model, self.tokenizer)
            msg.extend(msg_)
        return msg


def load_LLM(base_model_name, model_path, lora_path=None, vllm=True, gpu_memory_utilization=0.8):
    if vllm:
        return LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization)
    else:
        return MyLLM(base_model_name, model_path, lora_path)
