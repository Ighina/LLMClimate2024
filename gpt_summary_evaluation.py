import argparse
import json
import os
import sys

from datasets import load_dataset
import faiss
import nltk
import numpy as np
from openai import AzureOpenAI
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm
import wandb

from tools import load_config, search, azure_summarise

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
            '-b', '--bit4',
            action="store_true",
            help="Use 4 bit quantization when loading the LLM."
        )
        
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
            '-os', '--output_score',
            type=str,
            help="Where to store the final scores."
        )
        
        self.add_argument(
            '-p', '--prompt',
            default="Summarize the main takeaways from the following text",
            type=str,
            help="The basic prompt for querying the LLM."
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
            '-sump', '--summarise_partial',
            action="store_true",
            help="if included iteratively summarise paragraphs longer\
                 than given character threshold"
            )
            
        self.add_argument(
            '-cht', '--character_threshold',
            type=int,
            default=10000,
            help="if above option is included, this variable define\
                 the given character threshold over which to summarise\
                 the paragraphs individually before combining them."
            )
            
        self.add_argument(
            '-start', '--start_from',
            default=0,
            type=int,
            help="The document from which to start from. Use for checkpoint \
                  if you already run the code up to a certain document."
            )
            
        self.add_argument(
            '-w', '--wait',
            default=45,
            type=int,
            help="Number of seconds to wait after API call. This is \
                  used to avoid exceeding the call rate limit."
            )
        
        self.add_argument
        
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
    
    pricing = {"gpt-35-ghinzo": [0.0005/1000, 0.0015/1000], 
               "gpt-4-0125-ghinzo": [0.01/1000, 0.03/1000]}
    
    nltk.download("punkt")
    
    # add all the relevant environment variables before hand
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    model=args.model
    
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
    device = "cpu" if args.device=="cpu" else "cuda"
    evaluator = get_evaluator(task, device=device)
    prompt = args.prompt
    
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
    local_table = []
    out_file = args.output
    status = "success"
    tot_price = 0

    if args.start_from:
        local_table = pd.read_csv(out_file, index_col=0)
        local_table = local_table.values.tolist()
    
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
      
      if ex_n<args.start_from:
          assert len(local_table)==args.start_from, "The number of provided temporary results does not match the starting point of your current experiment."
          tmp = local_table[ex_n]
          table.add_data(
                  tmp[0],
                  args.model,
                  args.subset,
                  tmp[3],
                  tmp[4],
                  tmp[5],
                  tmp[6],
                  tmp[7],
                  tmp[8],
                  tmp[9],
                  tmp[10],
                  tmp[11]
              )
          wandb.log(
                   {
                    "coherence": tmp[6],
                    "consistency": tmp[7],
                    "fluency": tmp[8],
                    "relevance": tmp[9],
                    "overall": tmp[10]
                  }
                )
                
          all_coherence.append(tmp[6])
          all_consistency.append(tmp[7])
          all_fluency.append(tmp[8])
          all_relevance.append(tmp[9])
          all_overall.append(tmp[10])
                
          ex_n += 1
          continue
      
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
      
      reference = row["summary"]
      key = row["ID"]
      # print(key)
      
      # TODO: CREATE CLASS IN TOOLS TO DO API CALL
      # SHOULD HAVE DEPLOYMENT_NAME AS ATTRIBUTE

      start = time.time()
      if args.summarise_partial and len(question)>args.character_threshold:
          partial_question = []
          print(len(row["full_paragraphs"]))
          for question in row["full_paragraphs"]:
              print(question)
              summary, new_prompt, price = azure_summarise(
                                            client,
                                            model,
                                            question, 
                                            prompt, 
                                            argument,
                                            pricing)
              
              tot_price += price
              
              summary = response.choices

              partial_question.append(summary.strip())
          
          question = "\n".join(partial_question)
      
      summary, new_prompt, price = azure_summarise(client,
                                                      model,
                                                      question, 
                                                      prompt, 
                                                      argument,
                                                      pricing)
      print(status)
      tot_price += price
      print(f"Price so far: {tot_price}")
      
      seconds = time.time()-start
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
      
      if status=="success":
              local_table.append(
              [status,
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
              seconds]
              )
      else:
              local_table.append(
              [status,
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
              None]
              )
      
      pd.DataFrame(local_table, columns=columns).to_csv(out_file)
      time.sleep(args.wait)
    
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
    
    with open(args.output_score, "w") as f:
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
