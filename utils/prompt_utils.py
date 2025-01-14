import json
import os.path as osp
from typing import Union


# def get_pointwise_prompt(age, gender, diagnose, procedure, drug_candidate, model_name='llama-2-7b'):
#     """
#     generate prompt for pointwise model
#     :param age: patient age, int
#     :param gender: patient gender, str
#     :param diagnose: patient diagnose, list
#     :param procedure: patient procedure, list
#     :param drug_candidate: drug candidate, str
#     :return: prompt, str
#     """

#     # You are a professional doctor and you are experienced on prescribing medications.
#     # The answer is either <Yes.> or <No.> and do not explain.
#     if model_name == 'medalpaca-13b':
#         content = f"""
#                     You are taking a medical exam right now.
#                     You will be given a patient's clinical condition with a candidate drug,
#                     your task is to judge whether the candidate drug is effective and safe for the patient.
#                     Answer with <Yes.> or <No.> and do not provide any other words. Do not repeat the question in your answer.

#                     Age: {age},
#                     Gender: {gender},
#                     Diagnose: {diagnose},
#                     Procedure: {procedure}.
#                     Candidate drug: {drug_candidate}.

#                     Is the candicate drug effective and safe for the patient?:
#                     """
#     elif model_name == 'PMC_LLaMA_13B':
#         content = f"""
#                     You are taking a medical exam right now. \
#                     You will be given a patient's clinical condition with a candidate drug, \
#                     your task is to judge whether the candidate drug is effective and safe for the patient. \
#                     Answer with <Yes.> or <No.> and do not provide any other words.

#                     Age: {age},
#                     Gender: {gender},
#                     Diagnose: {diagnose},
#                     Procedure: {procedure}.
#                     Candidate drug: {drug_candidate}.

#                     Answer:
#                     """
#     else:     # LLAMA-2
#         content = f"""
#                     You are taking a medical exam right now.
#                     You will be given a patient's clinical condition with a candidate drug delimited by triple quotes,
#                     your task is to judge whether the candidate drug is effective and safe for the patient.
#                     Note: Answer with <Yes.> or <No.> and do not explain.
#                     '''
#                     Age: {age},
#                     Gender: {gender},
#                     Diagnose: {diagnose},
#                     Procedure: {procedure}.
#                     Candidate drug: {drug_candidate}.
#                     '''
#                     Answer:
#                     """
#     content = textwrap.dedent(content)
#     return content

# def get_evaluation_prompt(item_data, model_name='llama-2-13b-chat'):
#     (index, patient_info), drug_candidate = item_data
#     age = patient_info['AGE']
#     gender = patient_info['GENDER']
#     diagnose = patient_info['diagnose']
#     procedure = patient_info['procedure']

#     prompt = get_pointwise_prompt(age, gender, diagnose, procedure, drug_candidate, model_name)

#     return prompt

class Prompter(object):
    # __slots__ = ("template", "_verbose")   # 限制实例的属性

    def __init__(self, template_name: str = "", verbose: bool = False, use_history: bool = True, use_note: bool = True, use_disease: bool = True, use_procedure: bool = True):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "llama-2"
        file_name = osp.join("../utils", "templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name, encoding='utf-8') as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
        self.use_history = use_history
        self.use_note = use_note
        self.use_disease = use_disease
        self.use_procedure = use_procedure

    def generate_prompt(
        self,
        history: Union[None, str] = None,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.

        if self.use_history and history is not None:
            res = self.template["prompt_history"].format(
                history=history, input=input)
        else:
            res = self.template["prompt_no_history"].format(input=input)
        # res = self.template["prompt_input"].format(input=input)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

    def generate_history(self, patient_info):
        age = patient_info['AGE']
        gender = patient_info['GENDER']
        diagnose = patient_info['diagnose']
        procedure = patient_info['procedure']
        drug_name = patient_info['drug_name']
        
        note = None
        # if 'NOTE' in patient_info.keys():
        #     note = patient_info['NOTE'].strip('\n ')
        # else:
        #     note = None

        if self.use_note and note is not None:
            if self.use_disease and self.use_procedure:
                history = f'{note} ' \
                        f'Diagnose: {diagnose}, ' \
                        f'Procedure: {procedure}, ' \
                        f'Drug names: {drug_name}.'
            elif self.use_disease:
                history = f'{note} ' \
                        f'Diagnose: {diagnose}, ' \
                        f'Drug names: {drug_name}.'
            elif self.use_procedure:
                history = f'{note} ' \
                        f'Procedure: {procedure}, ' \
                        f'Drug names: {drug_name}.'
            else:
                history = f'{note} ' \
                        f'Drug names: {drug_name}.'
        else:
            if self.use_disease and self.use_procedure:
                history = f'Age: {age}, ' \
                        f'Gender: {gender}, '\
                        f'Diagnose: {diagnose}, ' \
                        f'Procedure: {procedure}, ' \
                        f'Drug names: {drug_name}.'
            elif self.use_disease:
                history = f'Age: {age}, ' \
                        f'Gender: {gender}, '\
                        f'Diagnose: {diagnose}, ' \
                        f'Drug names: {drug_name}.'
            elif self.use_procedure:
                history = f'Age: {age}, ' \
                        f'Gender: {gender}, '\
                        f'Procedure: {procedure}, ' \
                        f'Drug names: {drug_name}.'
            else:
                history = f'Age: {age}, ' \
                        f'Gender: {gender}, '\
                        f'Drug names: {drug_name}.'
                
        return history

    def generate_input(
        self,
        patient_info,
        drug_candidate,
    ):
        age = patient_info['AGE']
        gender = patient_info['GENDER']
        diagnose = patient_info['diagnose']
        procedure = patient_info['procedure']
        if 'NOTE' in patient_info.keys():
            note = patient_info['NOTE'].strip('\n ')
        else:
            note = None

        if self.use_note and note is not None:
            if self.use_disease and self.use_procedure:
                input_text = f'{note} ' \
                        f'Diagnose: {diagnose}, ' \
                        f'Procedure: {procedure}, ' \
                        f'Candidate drug: {drug_candidate}.'
            elif self.use_disease:
                input_text = f'{note} ' \
                        f'Diagnose: {diagnose}, ' \
                        f'Candidate drug: {drug_candidate}.'
            elif self.use_procedure:
                input_text = f'{note} ' \
                        f'Procedure: {procedure}, ' \
                        f'Candidate drug: {drug_candidate}.'
            else:
                input_text = f'{note} ' \
                        f'Candidate drug: {drug_candidate}.'
        else:
            if self.use_disease and self.use_procedure:
                input_text = f'Age: {age}, ' \
                            f'Gender: {gender}, ' \
                            f'Diagnose: {diagnose}, ' \
                            f'Procedure: {procedure}, ' \
                            f'Candidate drug: {drug_candidate}.'
            elif self.use_disease:
                input_text = f'Age: {age}, ' \
                            f'Gender: {gender}, ' \
                            f'Diagnose: {diagnose}, ' \
                            f'Candidate drug: {drug_candidate}.'   
            elif self.use_procedure:
                input_text = f'Age: {age}, ' \
                            f'Gender: {gender}, ' \
                            f'Procedure: {procedure}, ' \
                            f'Candidate drug: {drug_candidate}.'       
            else:
                input_text = f'Age: {age}, ' \
                            f'Gender: {gender}, ' \
                            f'Candidate drug: {drug_candidate}.'                        
        return input_text

    def get_evaluation_prompt(self, item_data):
        assert len(item_data) == 3, "item_data should be a tuple of (history, patient_info, drug_candidate)"

        history_info, patient_info, drug_candidate = item_data
        if history_info is not None:
            history = self.generate_history(history_info)
        else:
            history = None
        input_text = self.generate_input(patient_info, drug_candidate)
        prompt = self.generate_prompt(history=history, input=input_text)

        return prompt
