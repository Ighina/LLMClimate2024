import argparse
import json
import os
import sys

from datasets import load_dataset
import nltk
import numpy as np
import time
from tqdm import tqdm
import wandb

from tools import _load_model, load_config, summarise_question

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
            '-p', '--prompt',
            default="Summarize the main takeaways from the following text",
            type=str,
            help="The basic prompt for querying the LLM."
        )
            
        self.add_argument(
            '-s', '--subset',
            choices=["AR6", "AR5", "ALL"],
            default="ALL",
            help="The subset of the dataset to use."
        )
        
        # TODO: it appears that the model fails when going over the 
        # max model length. Try way to fix this (e.g. by iteratively
        # summarising the long paragraphs to be summarised)
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
        wandb.init(config=wandb_config)
        use_wandb = True
    except wandb.errors.UsageError:
        print("WARNING: NO WANDB KEY HAS BEEN SET! THE EXPERIMENT WILL BE LOGGED JUST LOCALLY!")
        os.environ["WANDB_DISABLED"] = "true"
        use_wandb = False
    
    # model instantiation
    model, tokenizer = _load_model(args.model, args.bit4)
    
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
    device = "cpu" # TODO: change this... How can both models fit?
    evaluator = get_evaluator(task, device=device)
    device = "cpu" if args.device=="cpu" else "cuda"
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

    for row in tqdm(data["test"]):
      
      question = "\n".join(row["full_paragraphs"])
      argument = row["summary_topic"]
      reference = row["summary"]
      key = row["ID"]
      # print(key)
      
      start = time.time()
      if args.summarise_partial and len(question)>args.character_threshold:
          partial_question = []
          print(len(row["full_paragraphs"]))
          for question in row["full_paragraphs"]:
              print(question)
              summary, new_prompt, status = summarise_question(model,
                                                       tokenizer,
                                                       question, 
                                                       prompt, 
                                                       argument,
                                                       device)
              
              partial_question.append(summary[len(new_prompt):])
          
          question = "\n".join(partial_question)
              
      
      
      summary, new_prompt, status = summarise_question(model,
                                                       tokenizer,
                                                       question, 
                                                       prompt, 
                                                       argument,
                                                       device)
      print(status)
      seconds = time.time()-start      
      summary = summary[len(new_prompt):]
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
