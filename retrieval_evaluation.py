import argparse
import json
import os
import sys

from datasets import load_dataset
import faiss
import nltk
import numpy as np
import pandas as pd
from ranx import Qrels, Run, evaluate
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, ndcg_score
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from tools import last_token_pool, _load_model, search

working_directory = os.getcwd()

####################################################################
# CUSTOM ARGUMENTS
####################################################################

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            description='InputOptions')
        
        self.add_argument(
            '-b', '--batch_size',
            type=int,
            default=32,
            help="The batch size to use when encoding \
                  documents.")
        
        self.add_argument(
            '-bit', '--bit_4',
            action="store_true",
            help="If included use the 4 bit quantized version \
                  of the given model. This option works just with \
                  transformers model, not sentence transformers."
            )
            
        self.add_argument(
            '-d', '--device',
            choices=["cpu", "gpu"],
            default="cpu",
            help="The device on which to run the experiment:\
                   cpu or gpu")
        
        self.add_argument(
            '-m', '--model',
            type=str,
            help="The LLM model to use.")
        
        self.add_argument(
            '-met', '--metric',
            choices=["recall", "IR"],
            default="recall",
            help="Whether to use the classic F1, precision and \
                  recall metrics (i.e. recall) or the Information \
                  Retrieval metrics map, mrr and ndcg, all at k=10."
            )
         
        self.add_argument(
            '-o', '--output',
            type=str,
            help="Where to store results."
        )
        
        self.add_argument(
            '-p', '--precision',
            default="float32",
            choices=["float32", "int8", "uint8", 
                     "binary", "ubinary"],
            help= "The type of precision to use when \
                   encoding the documents."
            )
        
        self.add_argument(
            '-rk', '--rag_top_k',
            default=2,
            type=int,
            help="The top documents to retrieve if doing RAG."
            )
            
        self.add_argument(
            '-s', '--subset',
            choices=["AR6", "AR5", "ALL"],
            default="ALL",
            help="The subset of the dataset to use."
        )
        
        self.add_argument(
            '-ut', '--use_transformer',
            action="store_true",
            help="If added, use transformers model with \
                  last token pooling, instead of sentence \
                  transformers. Use this option if model \
                  does not support sentence transformers \
                  and/or you want to use quantization."
            )
            
        self.add_argument(
            '-part', '--paragraph_topic',
            action="store_true",
            help="If the option is included, add the \
                  paragraph topic information when retrieving \
                  documents."
            )
            
        self.add_argument(
            '-sect', '--section_topic',
            action='store_true',
            help="If the option is included, add the \
                  section topic information when retrieving \
                  documents."
            )
        
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))

###################################################################
# MAIN FUNCTION
###################################################################

def main(args):
    
    nltk.download("punkt")
    
    # load the dataset
    data = load_dataset("sumipcc_dataset", args.subset)
    
    # initialize variables
    all_f1s = []
    all_precisions = []
    all_recalls = []
    ndcgs = []
    mrrs = []
    maps = []
    
    if args.metric=="IR":
        eval_metrics = ["precision@1",
                        "recall@1",
                        "precision@2",
                        "recall@2",
                        "precision@3",
                        "recall@3",
                        "precision@5",
                        "recall@5",
                        "precision@10",
                        "recall@10",
                        "map@10", 
                        "mrr@10", 
                        "ndcg@10"]
        
        eval_dict = {k:[] for k in eval_metrics}
    
    # Initialize evaluator for a specific task
    device = "cpu" if args.device=="cpu" else "cuda"
    
    if os.path.exists(args.output):
        local_table = pd.read_csv(args.output, index_col=0).values.tolist()
        with open(args.output.split(".")[0]+".json") as f:
            json_results = json.load(f)
    else:
        local_table = []
        json_results = {}
    
    json_results[args.model] = {}
    
    docs = set([])
    for doc in data["test"]:
        docs.update(doc["full_paragraphs"])
        
    docs = list(docs)
    
    if args.metric=="IR":
        ids = [f"q_{idx}" for idx in range(len(docs))]
    
    if args.use_transformer:
        model, tokenizer = _load_model(args.model,
                                       args.bit_4)
        try:
            model.to(device)
        except ValueError:
            pass            
        
        embeddings = []
        prev_idx = 0
        for idx in range(args.batch_size, len(docs), args.batch_size):
            batch_dict = tokenizer(docs[prev_idx:idx], 
                                   padding=True, 
                                   truncation=True,
                                   max_length=1000,
                                   return_tensors="pt")
            
            batch_dict = {k:v.to(device) for k, v in batch_dict.items()}
            print(batch_dict["input_ids"].shape)
                                   
            outputs = model(**batch_dict, output_hidden_states=True)
            embedding_b = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask']).detach().cpu().numpy()
            del outputs
            embeddings.append(embedding_b)
            prev_idx = idx
        if idx<len(docs): 
            batch_dict = tokenizer(docs[idx:], 
                                   padding=True, 
                                   truncation=True,
                                   max_length=1000,
                                   return_tensors="pt")
            
            batch_dict = {k:v.to(device) for k, v in batch_dict.items()}
                                   
            outputs = model(**batch_dict)
            embedding_b = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask']).detach().cpu().numpy()
            del outputs
            embeddings.append(embedding_b)
        
        embeddings = np.concatenate(embeddings, axis=0)
    else:
        encoding_model = SentenceTransformer(args.model, device=device, 
                                         trust_remote_code=True)
        
        embeddings = encoding_model.encode(docs, 
                                       batch_size=args.batch_size, 
                                       precision=args.precision)
        tokenizer = None
        
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, 'all_index')
    index = faiss.read_index('all_index')
    
    ex_n = 0
    
    for row in tqdm(data["test"]):
        
      argument = row["summary_topic"]
      
      if args.paragraph_topic:
          argument = ": ".join([row["paragraph_topic"], argument])
      
      if args.section_topic:
          argument = ": ".join([row["section_topic"], argument])
      
      retrieved_docs = search(argument,
                                  encoding_model,
                                  index, 
                                  docs,
                                  args.rag_top_k,
                                  tokenizer,
                                  args.metric=="IR")
      
      true_docs = row["full_paragraphs"]
      if args.metric=="recall":
      
          hits = [1 if doc in true_docs else 0 for doc in retrieved_docs]
      
          precision = sum(hits)/len(retrieved_docs)
      
          recall = sum(hits)/len(true_docs)
      
          try:
              f1 = (2*precision*recall)/(precision+recall)
          except ZeroDivisionError:
              f1 = 0
          
          all_precisions.append(precision)
          
          all_recalls.append(recall)
          
          all_f1s.append(f1)
          
          local_table.append([f1, precision, recall, args.model])
          
          if not ex_n % 20:
              print(precision)
              print(recall)
              print(f1)
      
      elif args.metric=="IR":
          qrel = {"query":{}}
          for k, v in zip(ids, docs):
              if v not in true_docs:
                  qrel["query"][k] = 0
              else:
                  qrel["query"][k] = 1
          qrel = Qrels(qrel)
          run = Run(retrieved_docs)
          
          results = evaluate(qrel, run, eval_metrics)
          
          tab_row = []
          for metric in eval_metrics:
              eval_dict[metric].append(results[metric])
              tab_row.append(results[metric])
          tab_row.append(args.model)
          local_table.append(tab_row)
    
    if args.metric=="recall":  
        pd.DataFrame(local_table, columns=["F1", "Precision", "Recall", "Model"]).to_csv(args.output)
    
        json_results[args.model] = {"f1":np.mean(all_f1s), 
                                "precision":np.mean(all_precisions), 
                                "recall":np.mean(all_recalls)}
    elif args.metric=="IR":
        pd.DataFrame(local_table, columns=eval_metrics+["Model"]).to_csv(args.output)
    
        json_results[args.model] = {k:np.mean(v) for k, v in eval_dict.items()}
                                
    with open(args.output.split(".")[0]+".json", "w") as f:
        json.dump(json_results, f)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args(sys.argv[1:])
    main(args)
