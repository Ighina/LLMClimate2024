import argparse
import json
import os
import sys

from datasets import load_dataset
import faiss
import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from tools import search

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
    
    # Initialize evaluator for a specific task
    device = "cpu" if args.device=="cpu" else "cuda"
    
    encoding_model = SentenceTransformer(args.model, device=device, 
                                         trust_remote_code=True)
    
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
    
    if args.use_transformer:
        model, tokenizer = _load_model(args.model,
                                       args.bit_4)
        batch_dict = tokenizer(docs, 
                               padding=True, 
                               truncation=True, 
                               return_tensors="pt")
                               
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = query_vector.detach().cpu().numpy()
    else:    
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
     
      retrieved_docs = search(argument,
                                  encoding_model,
                                  index, 
                                  docs,
                                  args.rag_top_k,
                                  tokenizer)
                                  
      true_docs = row["full_paragraphs"]
      
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
      
    pd.DataFrame(local_table, columns=["F1", "Precision", "Recall", "Model"]).to_csv(args.output)
    
    json_results[args.model] = {"f1":np.mean(all_f1s), 
                                "precision":np.mean(all_precisions), 
                                "recall":np.mean(all_recalls)}
    
    with open(args.output.split(".")[0]+".json", "w") as f:
        json.dump(json_results, f)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args(sys.argv[1:])
    main(args)
