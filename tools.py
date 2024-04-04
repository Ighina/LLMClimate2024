import os
import re
from typing import List, Dict, Callable, Set
import yaml

from jinja2.exceptions import TemplateError
from transformers import AutoModelForCausalLM, AutoTokenizer

def _load_model(model_name: str, bit4: bool = False) -> Callable:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=bit4)
    model.eval()

    return model, tokenizer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config")

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        configs = yaml.safe_load_all(file)
        list_config = []
        for config in configs:
            list_config.append(config)
    return list_config    

def summarise_question(
    model,
    tokenizer,
    question: str,
    prompt: str,
    argument: str,
    device: str="cpu",
    no_arg: bool=False,
    no_role: bool=False,
    max_len: int=10000
    ) -> str:
    """
    code mostly from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
  
    Parameters
    ----------
    model:    transformers.AutoModel
              the LLM to be used for answering the question 
    
    tokenizer:transformers.AutoTokenizer
              the tokenizer associated with the above LLM
              
    question: str
              the text to answer (text row from the test dataset)

    prompt:   str
              the prompt, i.e. how you ask the LLM to answer. LLMs are
              extremely sensitive to how they are asked to solve a task
              and different LLMs answer differently to the same prompt.

    device:   str
              the device to use: "cpu", for CPU, or "cuda", for GPU. The
              smallest LLMs (7B) need at least 16GB of GPU (i.e. a single
              A100 or 2 T4). If using Google Colab with a single T4, then,
              use CPU (which it should take anyway around 6 second per question).
              
    no_arg    bool
              If True, do not add the topic to the prompt. Use for ablation studies.
    
    no_role   bool
              If True, do not add the role to the prompt. Use for ablation studies.
    
    Returns
    --------
    decoded:  str
              the answer as output by the LLM.
              
    prompt:   str
              prompt after inclusion of additional elements (i.e. role and topic)
              
    status:   str
              one of "success" or "fail" according to whether to code run correctly
    """
    question = re.sub("\n", " ", question)
    
    if no_role and no_arg:
        messages = [
        {"role": "user", "content": f"{prompt}\nText: {question}"}
        ]
    elif no_role:
        messages = [
        {"role": "user", "content": f"{prompt} with respect to the topic: {argument}\nText: {question}"}
        ]
    elif no_arg:
        messages = [
        {"role": "system", "content": "You are an assistant to policy-makers.",
        "role": "user", "content": f"{prompt}\nText: {question}"}
        ]
    else:
        role_message = "You are an assistant to policy-makers."
        messages = [
        {"role": "system", "content": role_message},
        {"role": "user", "content": f"{prompt} with respect to the topic: {argument}\nText: {question}"}
        ]
    
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TemplateError:
        # if not supported, do not include system message
        prompt = tokenizer.apply_chat_template([messages[-1]], tokenize=False, add_generation_prompt=True)
    
    encodeds = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    
    prompt = tokenizer.batch_decode(encodeds)[0]
    
    model_inputs = encodeds.to(device)
    
    try:
        model.to(device)
    except ValueError:
        pass
    try:
        generated_ids = model.generate(model_inputs, max_new_tokens=5000, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids)[0]
        status = "success"
    except:
        decoded = prompt + "\nFAILED!"
        status = "fail"
        
    return decoded, prompt, status
