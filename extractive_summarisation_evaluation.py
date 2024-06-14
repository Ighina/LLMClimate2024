import argparse
import json
import os
import sys

from datasets import load_dataset
import faiss
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm
import wandb

from tools import _load_model, load_config, search, summarise_question

sys.path.insert(0, "UniEval")

from utils import convert_to_json
from metric.evaluator import get_evaluator

working_directory = os.getcwd()

####################################################################
# CUSTOM ARGUMENTS
####################################################################

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            description='InputOptions')
            
        self.add_argument(
            '-d', '--device',
            choices=["cpu", "gpu"],
            default="cpu",
            help="The device on which to run the experiment:\
                   cpu or gpu")
                   
        self.add_argument(
            '-rag', '--do_rag',
            action="store_true",
            help="Whether to do RAG instead of using ground \
                  truth documents from which to generate \
                  summaries."
            )
        
        self.add_argument(
            '-m', '--model',
            type=str,
            help="The Sentence Transformer encoder model to use.")
            
        self.add_argument(
            '-o', '--output',
            type=str,
            help="Where to store results."
        )
        
        self.add_argument(
            '-rm', '--rag_model',
            default="all-mpnet-base-v2",
            type=str,
            help="The name of the encoding model for retrieval to use if doing \
                  RAG: currently only supports SentenceTransformers models. "
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
            '-tk', '--top_k',
            default=2,
            type=int,
            help="The number of sentences to extract \
                as the summary of the given input."
        )
            
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))

###################################################################
# MAIN FUNCTION
###################################################################

def main(args):
    # initialize end of turn mark (dependent on model)
    #if args.model.startswith("google/gemma"):
    #    end_turn = "<end_of_turn>\n"
    #elif args.model.startswith("mistralai"):
    #    end_turn = "[/INST]"
    #elif args.model.startswith("microsoft")
    
    # initialize wandb
    
    nltk.download("punkt")
    
    wandb_config = load_config("config.yaml")[0]
    os.environ["WANDB_PROJECT"] = wandb_config["project"]
    try:
        os.environ["WANDB_API_KEY"] = wandb_config["key"]
        wandb.init(config=wandb_config, entity=wandb_config["entity"])
        use_wandb = True
    except wandb.errors.UsageError:
        print("WARNING: NO WANDB KEY HAS BEEN SET! THE EXPERIMENT WILL BE LOGGED JUST LOCALLY!")
        os.environ["WANDB_DISABLED"] = "true"
        use_wandb = False
    
    # model instantiation
    device = "cpu" if args.device=="cpu" else "cuda"

    model = SentenceTransformer(args.model, device=device)
    
    # load the dataset
    data = load_dataset("sumipcc_dataset", args.subset)
    
    # initialize variables
    all_coherence = []
    all_consistency = []
    all_fluency = []
    all_relevance = []
    all_overall = []
    
    all_summaries = []
    all_keys = []
    
    # Initialize evaluator for a specific task
    task = 'summarization'
    evaluator = get_evaluator(task, device=device)
    
    if use_wandb:
        columns = [
        "status",
        "model_name",
        "dataset",
        "identifier",
        "prompt",
        "response",
        "coherence",
        "consistency",
        "fluency",
        "relevance",
        "overall",
        "response_time_seconds",
        ]
        table = wandb.Table(columns=columns)
        wandb.run.log_code(".")
    
    ex_n = 0
    status = "success"
    new_prompt = ""
    
    if args.do_rag:
        encoding_model = SentenceTransformer(args.rag_model)
        
        docs = set([])
        for doc in data["test"]:
          docs.update(doc["full_paragraphs"])
        
        docs = list(docs)
        
        embeddings = encoding_model.encode(docs)
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, 'all_index')
        index = faiss.read_index('all_index')
    
    for row in tqdm(data["test"]):
      
        argument = row["summary_topic"]
        
        if args.do_rag:
            retrieved_docs = search(argument,
                                    encoding_model,
                                    index, 
                                    docs,
                                    args.rag_top_k)
                                    
            question = "\n".join(retrieved_docs)
        else:
            question = "\n".join(row["full_paragraphs"])
        
        sentences = []
        for sentence in nltk.tokenize.sent_tokenize(question):
            sentences.append(sentence)
        
        embeddings = model.encode(sentences)
        
        s_index = faiss.IndexFlatL2(embeddings.shape[1])
        s_index.add(embeddings)
        faiss.write_index(s_index, 'all_sentences')
        s_index = faiss.read_index('all_sentences')
        
        reference = row["summary"]
        key = row["ID"]
        # print(key)
        
        start = time.time()
        retrieved_docs = search(argument,
                                model,
                                s_index, 
                                sentences,
                                args.top_k)
        print(status)
        seconds = time.time()-start      
        summary = "\n".join(retrieved_docs).strip()
        if not ex_n%20:
            print("Example output:\n")
            print(new_prompt)
            print(summary)
        all_summaries.append(summary)
        all_keys.append(key)

        # Prepare data for pre-trained evaluators
        data_json = convert_to_json(output_list=[summary],
                            src_list=[question],
                            ref_list=[reference])
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data_json, print_result=True)
        
        coherence = eval_scores[0]["coherence"]
        consistency = eval_scores[0]["consistency"]
        fluency = eval_scores[0]["fluency"]
        relevance = eval_scores[0]["relevance"]
        overall = eval_scores[0]["overall"]
        
        all_coherence.append(coherence)
        all_consistency.append(consistency)
        all_fluency.append(fluency)
        all_relevance.append(relevance)
        all_overall.append(overall)
        
        ex_n += 1
        
        # log results to wandb (if using)
        if use_wandb:
            if status=="success":
                table.add_data(
                status,
                args.model,
                args.subset,
                key,
                new_prompt,
                summary,
                coherence,
                consistency,
                fluency,
                relevance,
                overall,
                seconds
                )
                wandb.log(
                        {
                        "coherence": coherence,
                        "consistency": consistency,
                        "fluency": fluency,
                        "relevance": relevance,
                        "overall": overall
                        }
                        )
            else:
                table.add_data(
                status,
                args.model,
                args.subset,
                key,
                new_prompt,
                None,
                None,
                None,
                None,
                None,
                None,
                None
                )
    
    mean_coherence = np.mean(all_coherence)
    mean_consistency = np.mean(all_consistency)
    mean_fluency = np.mean(all_fluency)
    mean_relevance = np.mean(all_relevance)
    mean_overall = np.mean(all_overall)
    
    if use_wandb:
        # Set summary value for the line plots to be the mean overall scores
        # Otherwise these are recorded as the final scores
        wandb.run.summary["coherence"] = mean_coherence
        wandb.run.summary["consistency"] = mean_consistency
        wandb.run.summary["fluency"] = mean_fluency
        wandb.run.summary["relevance"] = mean_relevance
        wandb.run.summary["overall"] = mean_overall
        
        wandb.log({"coherence_avg":mean_coherence})
        wandb.log({"consistency_avg":mean_consistency})
        wandb.log({"fluency_avg":mean_fluency})
        wandb.log({"relevance_avg":mean_relevance})
        wandb.log({"overall_avg":mean_overall})
        wandb.log({"Summarisation Results": table})
        
    with open(args.output, "w") as f:
        json.dump({"coherence":mean_coherence,
                   "consistency":mean_consistency,
                   "fluency": mean_fluency,
                   "relevance": mean_relevance,
                   "overall": mean_overall
        }, f)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args(sys.argv[1:])
    main(args)