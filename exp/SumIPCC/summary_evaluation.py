import argparse
import os
import sys

from datasets import load_dataset
import wandb

from unieval.utils import convert_to_json
from unieval.metric.evaluator import get_evaluator

from utils import _load_model, summarise_question

working_directory = os.getcwd()

####################################################################
# CUSTOM ARGUMENTS
####################################################################

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            description='InputOptions',
            formatter_class=RawTextHelpFormatter)
            
        self.add_argument(
            '-d', '--device',
            choice=["cpu", "gpu"],
            default="cpu",
            help="The device on which to run the experiment:\
                   cpu or gpu")
                   
        self.add_argument(
            '-b', '--4bit',
            action="store_true".
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
            choice=["AR6", "AR5", "ALL"],
            default="ALL",
            help="The subset of the dataset to use."
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
    wandb_config = load_config("config.yaml")
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
    model, tokenizer = _load_model(args.model, args.4bit)
    
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
    evaluator = get_evaluator(task, device=args.device)
    
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

    for row in data["test"]:
      
      question = "\n".join(data["full_paragraphs"])
      argument = row["summary_topic"]
      reference = row["summary"]
      
      summary, new_prompt, status = summarise_question(question, 
                                                       prompt, 
                                                       argument)
          
      summary = summary[len(new_prompt)+1:]
      all_summaries.append(summary)
      all_keys.append(key)

      # Prepare data for pre-trained evaluators
      data_json = convert_to_json(output_list=[summary],
                            src_list=[question],
                            ref_list=[reference])
      # Get multi-dimensional evaluation scores
      eval_scores = evaluator.evaluate(data_json, print_result=True)
      
      coherence = eval_scores[0]["coherence"]
      consistency = eval_scores[0]["constistency"]
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
              overall
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
              None
              )
    
    mean_coherence = np.mean(coherence)
    mean_consistency = np.mean(consistency)
    mean_fluency = np.mean(fluency)
    mean_relevance = np.mean(relevance)
    mean_overall = np.mean(overall)
    
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
    
if __name__ == '__main__':
    parser = PredictParser()
    args = parser.parse_args(sys.argv[1:])
    main()
