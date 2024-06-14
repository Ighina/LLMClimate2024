import os
import re
from typing import List, Dict, Callable, Set
import yaml

from jinja2.exceptions import TemplateError
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

def create_run(top_k):
  run = {"query":{f"q_{idx}":1-score for idx, score in zip(top_k[1][0],top_k[0][0])}}
  return run

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def _load_model(model_name: str, bit4: bool = False) -> Callable:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=bit4, trust_remote_code=True)
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

def search(query: str,
           model,
           index, 
           collection: List[str], 
           k: int=2,
           tokenizer = None,
           rankx: bool = False) -> str:
    """
    Parameters
    ----------
    query:      str
                the query to retrieve the document in the index
    model:      sentence_transformers.SentenceTransformer
                the encoding model to encode the query. Should be the same 
                used for the faiss index. Currently we only support 
                SentenceTransformer
    index:      faiss.Index
                the faiss index where from which to retrieve the documents
    collection: list
                the list containing the documents in the index
    k:          int
                how many documents to retrieve
    tokenizer   transformers.Tokenizer
                if using transformer model, then add tokenizer to 
                encode the sentences via the transformers API
    rankx       bool
                If true, return the result in the rankx format
    Returns
    --------
    retrieval:  list
                the retrieved documents from the index
    """
    k = len(collection) if rankx else k # when evaluating get all the results
    if tokenizer is None:
        query_vector = model.encode([query])
    else:
        batch_dict = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        batch_dict = {k:v.to(model.device) for k, v in batch_dict.items()}
        
        outputs = model(**batch_dict)
        query_vector = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        query_vector = query_vector.detach().cpu().numpy()
    top_k = index.search(query_vector, k)  # top3 only
    if rankx:
        return create_run(top_k)
    return [collection[_id] for _id in top_k[1].tolist()[0]]

def summarise_question(
    model,
    tokenizer,
    question: str,
    prompt: str,
    argument: str,
    device: str="cpu",
    no_arg: bool=False,
    no_role: bool=True,
    dir_gen: bool=False,
    max_len: int=5000
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
              
    dir_gen   bool
              If True, does not add any additional element to the prompt (ignores \
              the "question" argument)
    
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
    
    if dir_gen:
        messages = [
        {"role": "user", "content": prompt}
        ]
    elif no_role and no_arg:
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
        if no_role:
          # Qwen 1.5 automatically adds the role, if you don't want it: delete it
          prompt = re.sub("<\|im_start\|>system\nYou are a helpful assistant<\|im_end\|>\n",
                          "",
                          prompt)
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
        generated_ids = model.generate(model_inputs, max_new_tokens=max_len, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids)[0]
        status = "success"
    except:
        decoded = prompt + "\nFAILED!"
        status = "fail"
        
    return decoded, prompt, status

def azure_summarise(
    client,
    model: str,
    question: str,
    prompt: str,
    argument: str,
    pricing: Dict[str, List],
    no_arg: bool=False,
    no_role: bool=True,
    max_len: int=10000
    ) -> str:
    """
    code mostly from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
  
    Parameters
    ----------
    client    openai.AzureOpenAI
              the instance of the Azure client class from which the API
              call is made.
              
    model:    str
              the model name, among the ones available from Azure API
              
    question: str
              the text to answer (text row from the test dataset)

    prompt:   str
              the prompt, i.e. how you ask the LLM to answer. LLMs are
              extremely sensitive to how they are asked to solve a task
              and different LLMs answer differently to the same prompt.

    pricing:  dict
              Dictionary of the form {"model_name":[input_price, output_price]}
              for tracking the overall spenditure.
              
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
    
    price:    float
              the total price of the API call
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
    
    response = client.chat.completions.create(
    model=model,
    messages=messages)
    
    prompt = " ".join([msg["content"] for msg in messages])
    
    decoded = response.choices[0].message.content
    
    in_price = pricing[model][0]*response.usage.prompt_tokens
    out_price = pricing[model][0]*response.usage.completion_tokens
    
    price = in_price + out_price
    
    return decoded, prompt, price