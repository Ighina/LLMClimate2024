# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:04:08 2024

@author: User
"""
import argparse
import json
import re
import sys
from typing import List, Dict, Tuple
import yaml


###########################################################
# Custom Parser
###########################################################

class CustomParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="This script reads a yaml file \
                                    containing the dataset and postprocess it.")
                                     
        self.add_argument(
            "--dataset",
            "-d",
            type=str,
            help="The path to the dataset in yaml format."
            )
            
        self.add_argument(
            "--output",
            "-out",
            type=str,
            help="The path to the yaml file to write the processed dataset to."
            )
            
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)
        
        
###########################################################
# Functions
###########################################################

def postprocess_fn(dataset: 
                   Dict[str, Dict]) -> Tuple[Dict, List]:
    """Collect all the pointers in the dataset and return the incorrect keys
    Parameters
    ----------
    dataset:  dict[dict]
              the complete dataset
    Returns
    ---------
    incorrect Dict[Dict]
              the list of keys which have pointers that point to paragraphs which are
              incorrect either because they are too broad or beacuse they are shorter
              or just marginally longer than the summaries. These two cases are divided
              by key in the outcput dictionary, the first list under "long" and the second
              list under "short".
    sharing   Dict[List]
              a Dict of list of keys sharing the same paragraphs (and that are not in
              the incorrect category). Each group of key sharing the same paragrpah is
              aggregated in the same list, identified by the shared pointer
    """
    incorrect = {"long":[], "short":[]}
    
    max_lim_div = 100
    min_lim_div = 3/4 # we want summaries at most 3/4 the length of the original paragraph 
    # Find the summaries referring to too long paragraphs
    for key, paragraphs in dataset["full_paragraphs"].items():
        par_len = len(" ".join(paragraphs))
        if par_len>len(dataset["summaries"][key])*max_lim_div:
            incorrect["long"].append(key)
        elif par_len*min_lim_div<len(dataset["summaries"][key]):
            incorrect["short"].append(key)
    
    # Find all the summaries sharing a paragraph
    sharing = {}
    seen_pointers = set()
    for key, pointers in dataset["pointers"].items():
        if key in incorrect["long"]:
            continue 
        pointers = "$".join(pointers)
        if pointers not in seen_pointers:
            sharing[pointers] = [key]
            seen_pointers.add(pointers)
        else:
            sharing[pointers].append(key)
    # filter out the pointers referring to just one summary
    sharing_tmp = {k:v for k, v in sharing.items() if len(v)>1}
    sharing = {}
    
    for k, v in sharing_tmp.items():
        key = v[0]
        par_len = len(" ".join(dataset["full_paragraphs"][key]))
        summary = ""
        for sm in v:
            summary = " ".join([summary, dataset["summaries"][sm]])
        
        if par_len*min_lim_div<len(summary):
            for sm in v:
                incorrect["short"].append(sm)
        else:
            sharing[k] = v
            
    return incorrect, sharing
    
def aggregate_keys(dataset: 
                   Dict[str, Dict],
                   keys2aggregate:
                   Dict[str, List])->Dict[str, Dict]:
    """Aggregate keys sharing the same paragraph
    Parameters
    ----------
    dataset:        Dict[Dict]
                    the original dataset
    keys2aggregate: List[List]
                    the keys which shares the same paragraphs
    Returns
    ----------
    new_dataset Dict[Dict]
                the new dataset where the keys sharing same paragraphs
                have been aggregated
    """
    new_dataset = dataset.copy()
    
    for keys in keys2aggregate.values():
        keys = list(keys)
        main_key = keys[0]
        summary = new_dataset["summaries"][main_key]
        for key in keys[1:]:
            print(f"Merging summary {key} with summary {main_key} as they refer to the same target paragraph!")
            summary = " ".join([summary, new_dataset["summaries"][key]])
            for column in new_dataset:
                new_dataset[column].pop(key)
        new_dataset["summaries"][main_key] = summary
    
    return new_dataset
        
        
###########################################################
# Main
###########################################################

def main(args):
    with open(args.dataset) as f:
        data = yaml.safe_load(f)
    
    incorrect, sharing = postprocess_fn(data)
    
    for err_type in incorrect:
        for key in incorrect[err_type]:
            print(f"Deleting {key} from dataset because target paragraph is too {err_type}!")
            for col in data:
                data[col].pop(key)
            
    new_data = aggregate_keys(data, sharing)
    
    with open(args.output, "w") as f:
        yaml.safe_dump(new_data, f, 
                       encoding = "utf-8",
                       allow_unicode = True)
        

###########################################################
# Run
###########################################################

if __name__ == "__main__":
    parser = CustomParser()
    args = parser.parse_args(sys.argv[1:])
    main(args)
