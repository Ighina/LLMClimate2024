# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:04:08 2024

@author: User
"""
import argparse
import json
import re
import sys
from typing import List, Dict
import yaml


###########################################################
# Custom Parser
###########################################################

class CustomParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="This script reads three input documents:\
                         one including the summary for policy makers, one\
                             including the complete synthesis report and one\
                                 including a dictionary of topics associated\
                                     with the summaries.")
                                     
        self.add_argument(
            "--full_report",
            "-full",
            type=str,
            help="The path to the full report, having each paragraph separated\
                by an empty line and with paragraph number at the beginning."
            )
            
        self.add_argument(
            "--summaries",
            "-sum",
            type=str,
            help="The path to the summaries for policy makers, having each \
                paragraph separated\
                by an empty line and with summaries number at the beginning."
            )

        self.add_argument(
            "--topics",
            "-top",
            type=str,
            help="The path to the topics associated with the summaries for \
                policy makers. This should be a json with topics associated\
                    with the identifiers of the summaries as keys."
            )
            
        self.add_argument(
            "--report_style",
            "-rs",
            type=str,
            default="AR6",
            help="The report style, defining the characteristics of how\
            the report is structured in terms of identifiers and\
            paragraphs numbering."
        )
            
        self.add_argument(
            "--output",
            "-out",
            type=str,
            help="The path to the yaml file to write the dataset to."
            )
            
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)
        
        
###########################################################
# Functions
###########################################################

def clean(line: str) -> str:
    """process the input line to delete a variety of unwanted elements
    Parameters
    ----------
    line : str
           the line to process

    Returns
    -------
    line : str
           The processed line
    """
    # delete notes
    line = re.sub("([a-z\,])\d+", "\g<1>", line)
    # delete anything inside parenthesis
    line = re.sub("\(.*?\)", "", line)
    # delete any trailing space and realign punctuation
    line = re.sub("  ", " ", line)
    line = re.sub("(\w) ([\,\.\:\?\!;])", "\g<1>\g<2>", line)
    line = line.strip()
    
    #line = re.sub("\-$", "$", line)
    
    return line

def expand_pointers(pointers: 
                    str) -> List[str]:
    """Process pointer to bring it to correct format
    Parameters
    ----------
    pointers: str
              The input pointer
    Returns
    ---------
    points    List[str]
              the new processed pointers, i.e. a list of correct pointers
    """
    points = []
    
    pointer_list = pointers.split(", ")
    
    for pointer in pointer_list:
        # case in which it is pointing to something else than a paragraph
        if not re.findall("^\d", pointer):
            continue
        # case in which there is a range, e.g. 4.1 to 4.5
        if re.findall(" to ", pointer):
            range_p = pointer.split(" to ")
            # delete final dot so it's easier to process
            range_p = [re.sub("\.$", "", p) for p in range_p]
            
            points.append(range_p[0])
            points.append(range_p[1])
            
            start = int(range_p[0][-1])
            end = int(range_p[1][-1])
            
            # now also add all the paragraphs in between
            for n in range(start+1, end):
                points.append(range_p[0][:-1]+str(n))
        else:
            points.append(pointer)
            
    return points
    
def read_report(lines: 
                List[str],
                full: bool=False,
                summaries: bool=False,
                pattern: str="^[A-z\d]+\.(?:\d[\.]?)+")->Dict[str, str]:
    """Read a report, with empy lines separating paragraphs
    Parameters
    ----------
    lines:     list[str]
               the lines from the report's text file
    full:      bool
               if True, extract also the title next to the 
               paragraphs identifier separately (this works
               just for the full report's paragraphs')
    summaries  bool
               if True, indicates that we are processing
               the summaries instead of the full report (
                   and we apply the relative routine).
    pattern    str
               the pattern to apply for extracting the summary
               identifiers. Varies according to the report used
    Returns
    ----------
    paragraphs dict[str]
               the report divided into its componing paragraphs
               where the letters/numbers at the beginning of the 
               paragraphs are used as keys associated with  the
               paragraphs themselves as values
    """
    
    paragraphs = dict()
    current_paragraph = ""
    first_line = True
    hyphen = False
    for idx, line in enumerate(lines):
        if line.strip():
            if first_line:
                print(idx)
                first_line = False
                try:
                    identifier = re.findall(pattern, line)[0]
                except IndexError:
                    identifier = re.findall("(?:\d[\.]?)+", line)[0] 
                line = line[len(identifier)+1:]
                if full:
                    title = line[:]
                current_paragraph = line
            else:
                line = clean(line)
                if hyphen:
                    hyphen = False
                    current_paragraph = "".join([current_paragraph[:-1], line])
                elif line:
                    current_paragraph = " ".join([current_paragraph, line])
                if line.endswith("-"):
                    hyphen = True
        else:
            first_line = True
            hyphen = False
            paragraphs[identifier] = dict()
            if full:
                paragraphs[identifier]["title"] = title
            elif summaries:
                pointer = re.findall("\{([\w\.\,\s\-|:]+)\}", current_paragraph)[-1]
                paragraphs[identifier]["pointer"] = pointer
            paragraphs[identifier]["paragraph"] = re.sub("\{.*?\}", "", 
                                                         current_paragraph).strip()
            current_paragraph=""
    return paragraphs
        
        
###########################################################
# Main
###########################################################     

def main(args):
    with open(args.full_report, encoding="utf-8") as f:
        full_report = f.readlines()
        
    with open(args.summaries, encoding="utf-8") as f:
        summaries = f.readlines()
        
    with open(args.topics, encoding="utf-8") as f:
        topics = json.load(f)
        
    if args.report_style=="AR5":
        new_topics = {}
        for key in topics:
            new_key = re.sub("SPM ", "", key)
            new_topics[new_key] = topics[key]
        topics = new_topics
        print(topics)
        
        pattern = "SPM \d(?:\.\d)*[a-s]?"
    else:
        pattern = "^[A-z\d]+\.(?:\d[\.]?)+"
        
    full_report = read_report(full_report, full=True)
    
    summaries = read_report(summaries, summaries=True, pattern=pattern)
    
    final_data = dict(full_paragraphs=dict(), 
                      summaries=dict(),
                      titles=dict(),
                      paragraph_topics=dict(),
                      summary_topics=dict(),
                      section_topics=dict(),
                      pointers=dict()
                      )
    
    for key, summary in summaries.items():
        key = re.sub("SPM ", "", key)
        final_data["summaries"][key] = summary["paragraph"]
        
        final_data["paragraph_topics"][key] = topics[key[:3]]
        final_data["summary_topics"][key] = topics[key]
        final_data["section_topics"][key] = topics[key[0]]
            
        all_paragraphs = []
        all_titles = []
        
        pointers = expand_pointers(summary["pointer"])
        for pointer in pointers:
            if re.findall("^\d", pointer):
                if pointer not in full_report:
                    pointer+="."
                all_paragraphs.append(full_report[pointer]["paragraph"])
                all_titles.append(full_report[pointer]["title"])
        final_data["full_paragraphs"][key] = all_paragraphs
        final_data["titles"][key] = all_titles
        final_data["pointers"][key] = pointers
    
    with open(args.output, "w") as f:
        yaml.safe_dump(final_data, f, encoding='utf-8', allow_unicode=True)
        

###########################################################
# Run
###########################################################   

if __name__ == "__main__":
    parser = CustomParser()
    args = parser.parse_args(sys.argv[1:])
    main(args)
