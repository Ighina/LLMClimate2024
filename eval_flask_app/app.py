import os
import random

from datasets import load_dataset
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)

random.seed(40)

###################################################
# TODO: Will change this part with WANDB Table
###################################################



df = pd.read_csv("wandb_results.csv")

sample_ids = sample_ids = df["identifier"].sample(10, random_state = 40)

sample_df = df[df.identifier.isin(sample_ids)]

os.chdir("..")

data = load_dataset("sumipcc_dataset", "ALL")

references = {}
queries = {}

keys_to_use = ["1.1c", "1.4a", "2.2g", "A.1.5", "B.5.1",
               "C.2.4", "A.2.4", "A.1.5", "2.4b", "2.3e"]

for d in data["test"]:
    for ident in keys_to_use:
    #for ident in sample_ids:
        if d["ID"]==ident:
            references[ident] = d["summary"]
            topic = d["summary_topic"]
            query = "\n".join(d["full_paragraphs"])
            queries[ident] = f"Topic: {topic}.\n Text: {query}"

os.chdir("eval_flask_app")

all_available_models = pd.unique(sample_df.model_name)

available_summaries = {}
#queries = {}

models_per_key = [["Qwen/Qwen1.5-0.5B-Chat", "gpt-4-0125-ghinzo"],
                  ["mistralai/Mistral-7B-Instruct-v0.2", "Qwen/Qwen1.5-0.5B-Chat"],
                  ["meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen1.5-0.5B-Chat"],
                  ["google/gemma-1.1-2b-it", "gpt-35-ghinzo"],
                  ["gpt-35-ghinzo", "mistralai/Mistral-7B-Instruct-v0.2"],
                  ["meta-llama/Meta-Llama-3-8B-Instruct","mistralai/Mistral-7B-Instruct-v0.2"],
                  ["gpt-4-0125-ghinzo", "Qwen/Qwen1.5-0.5B-Chat"],
                  ["meta-llama/Meta-Llama-3-8B-Instruct", "gpt-35-ghinzo"],
                  ["Qwen/Qwen1.5-0.5B-Chat", "microsoft/Phi-3-mini-128k-instruct"],
                  ["Qwen/Qwen1.5-0.5B-Chat", "microsoft/Phi-3-mini-128k-instruct"]
                 ] 

for models, key in zip(models_per_key, keys_to_use):
    available_summaries[key] = []
    
    summary = references[key]
    #prompt = sample_df[sample_df.identifier==key].prompt.values[0]
    # queries[key] = prompt 
    #for model in all_available_models:
    #    available_summaries[key].append({model: sample_df[(sample_df.identifier==key)&(sample_df.model_name==model)].response.values[0]})
    for model in models:
        available_summaries[key].append({model: df[(df.identifier==key)&(df.model_name==model)].response.values[0].split("<")[0]})

#available_summaries = {
#    "key1":[{"model1": "Summary A"}, {"model2": "Summary B"}, {"model3": "Summary C"}, {"model4": "Summary D"}, {"model5": "Summary E"}],
#    "key2":[{"model1": "Summary A"}, {"model2": "Summary B"}, {"model3": "Summary C"}, {"model4": "Summary D"}, {"model5": "Summary E"}],
#    "key3":[{"model1": "Summary A"}, {"model2": "Summary B"}, {"model3": "Summary C"}, {"model4": "Summary D"}, {"model5": "Summary E"}],
#    "key4":[{"model1": "Summary A"}, {"model2": "Summary B"}, {"model3": "Summary C"}, {"model4": "Summary D"}, {"model5": "Summary E"}],
#    "key5":[{"model1": "Summary A"}, {"model2": "Summary B"}, {"model3": "Summary C"}, {"model4": "Summary D"}, {"model5": "Summary E"}],
#}

#queries = {"key1": "query1",
#           "key2": "query2",
#           "key3": "query3",
#           "key4": "query4",
#           "key5": "query5"} 

###################################################
# TODO: Will change above part with WANDB Table
###################################################

# Dictionary to store summaries and their votes
summaries = {
    "Coherence 1": 0,
    "Coherence 2": 0,
    "Fluency 1": 0,
    "Fluency 2": 0,
    "Relevance 1": 0,
    "Relevance 2": 0,
    "Consistency 1": 0,
    "Consistency 2": 0
}

key = None
model1 = None
model2 = None

evaluation_results = {}

aspects = ["Coherence", "Fluency", "Relevance", "Consistency"]

def get_random_summaries():
    # Randomly select two summaries from the available list
    
    key = random.sample(list(available_summaries.keys()), 1)[0]
    
    available_summs = available_summaries.pop(key)
    sum1, sum2 = random.sample(available_summs, 2)
    return key, sum1, sum2

@app.route('/', methods=['GET', 'POST'])
def index():
    global model1
    global model2
    global key
    
    if request.method == 'POST':
        
        global summaries
        summaries = {
    "Coherence 1": 0,
    "Coherence 2": 0,
    "Fluency 1": 0,
    "Fluency 2": 0,
    "Relevance 1": 0,
    "Relevance 2": 0,
    "Consistency 1": 0,
    "Consistency 2": 0
}
        for aspect in aspects:
            selected_summary = request.form[aspect.lower()]
            summaries[selected_summary] = 1
        
        evaluation_results[key] = {}
        
        for aspect in aspects:
            for model in all_available_models:
                 if model==model1:
                     evaluation_results[key][f"{model}_{aspect}"]=summaries[f"{aspect} 1"]
                 elif model==model2:
                     evaluation_results[key][f"{model}_{aspect}"]=summaries[f"{aspect} 2"]
                 else:
                     evaluation_results[key][f"{model}_{aspect}"]=-1
        
        return redirect(url_for('index'))
    
    # Get random summaries
    
    if not available_summaries:
        ids, results = [], []
        
        for key, value in evaluation_results.items():
            ids.append(key)
            results.append(value)
        
        pd.DataFrame(list(results), index=list(ids)).to_csv("evaluation_results.csv")
        
        return render_template("index2.html")
    
    key_new, summary_1, summary_2 = get_random_summaries()
    key = key_new
    model_uno, value1 = list(summary_1.items())[0]
    model_due, value2 = list(summary_2.items())[0]
    
    query = queries[key]
    
    reference = references[key]
    
    model1 = model_uno
    
    model2 = model_due
    
    return render_template('index.html', summary_1=value1, summary_2=value2, query=query, reference=reference)

if __name__ == '__main__':
    app.run(debug=True, port=8001)

