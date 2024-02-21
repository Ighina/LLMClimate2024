import json
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
from string2string.metrics import ROUGE, sacreBLEU
from string2string.similarity import BERTScore, BARTScore
from bleurt import score
import pandas as pd
import argparse
import statistics
import os
import sys
working_directory = os.getcwd()

# class named ArgumentParser for parsing options and handling errors
class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            description='InputOptions',
            formatter_class=RawTextHelpFormatter)
        self.add_argument(
            '-g', '--gen',
            type=str,
            help='''
            Provide json file containing generated summaries
            as produced by the get_segment_metadata script.
            This should be in the following format:
            [{
                "pid": "m0000d2q",
                "segments": [
                    {
                    "timestamps": {
                    "start": 0,
                    "end": 111.14
                    },
                    "title": "Introduction",
                    "summary": "The conductor, an immense ...",
                    "tags": [ "Music", ...]
                    ],
                "transcript": "Presenting the second Concerto Final...
                },
                {
                    ...
                }, ...
                ],
                "overall_summary": " ... ",
                "model_used": "gpt4"
            }]
                ''')
        self.add_argument(
            '-po', '--getpipsonly',
            action='store_true',
            help='''
            This tag must be used alongside the '--gen' tag.
            Using this will mean the run will stop short of evaluating
            the generated summaries, and instead will only print a file
            containing the PIPs summaries, as well as a file containing
            generated summaries alongside their corresponding
            PIPs summaries.
            ''')
        self.add_argument(
            '-p', '--pips',
            type=str,
            help='''
            Takes a json file containing PIPs summaries in the following
            format:
            [{
                "EpisodePid":"m0000d2q",
                "PIPsSynopsis":"The presenter meets conductor...."
                },
                { ... }, ...
            ]
            To be used alongside the '--gen' tag, and both must contain
            the same EpisodePid values.
            ''')
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))


def main():
    # Change the main!
    argParser = ArgumentParser()
    args = argParser.parse_args()

    gen_synopses = pd.DataFrame()
    pips_synopses = pd.DataFrame()

    if args.gen:
        print("Reading generated summaries from file " + str(args.gen))
        gen_synopses = pd.read_json(args.gen)
        gen_synopses = gen_synopses.rename(columns={'pid': 'EpisodePid'}, errors='ignore')
        gen_synopses = gen_synopses.rename(columns={'gen_synopses': 'GeneratedSynopsis'}, errors='ignore')
        gen_synopses = gen_synopses.rename(columns={'overall_summary': 'GeneratedSynopsis'}, errors='ignore')
        gen_synopses = gen_synopses.rename(columns={'overall_synopsis': 'GeneratedSynopsis'}, errors='ignore')
        gen_synopses = gen_synopses[['EpisodePid', 'GeneratedSynopsis']]
    else:
        argParser.error("No generated summaries provided. Use either --gen tag to input a file.")

    if args.pips:
        print("Reading PIPs summaries from file " + str(args.pips))
        pips_synopses = pd.read_json(args.pips)
    else:
        print("No PIPs summaries provided. Looking up episode pids in PIPs Inspector.")
        list_of_pids = gen_synopses['EpisodePid'].tolist()
        print("List of pids to look up in PIPs = " + str(list_of_pids))
        pips_synopses = extract_pips_synopses(list_of_pids)
        pips_synopses = pips_synopses.rename(columns={'EpisodePid': 'EpisodePid', 'synopsis': 'PIPsSynopsis'}, errors='ignore')
        pips_synopses = pips_synopses[['EpisodePid', 'PIPsSynopsis']]
        print("Printing PIPs summaries to " + str(args.gen[:-6] + str("_PIPs.jsonl")))
        pips_synopses.to_json(args.gen[:-6] + str("_PIPs.jsonl"),orient='records')

    if not args.getpipsonly:
        print("Evaluating summaries.")
        combined = pips_synopses.merge(gen_synopses, how='inner', on='EpisodePid')
        # output_filename = args.gen[:-6] + str("_PIPsGenCombined.jsonl")
        # print("Printing to file containing PIPs and generated summaries to " + output_filename)
        # combined.to_json(output_filename,orient='records')

        metrics = get_evaluation_metrics(combined)

        output_filename = args.gen[:-6] + str("_evaluationMetrics.json")
        print("Printing metrics to file " + output_filename)
        with open(output_filename, 'w') as outfile:
                json.dump(metrics, outfile)


def get_evaluation_metrics(combined):

    generated_summaries = combined['GeneratedSynopsis'].tolist()
    reference_summaries = combined['PIPsSynopsis'].tolist()
    episode_pids = combined['EpisodePid'].tolist()

    # Convert 'reference_summaries' to list of lists for ROUGE, BLEU and BERTscore
    reference_summaries_lists = combined['PIPsSynopsis']
    for index, row in combined.iterrows():
        reference_summaries_lists[index] = [row['PIPsSynopsis']]

    rogue_scorer = ROUGE()
    rouge_f1_result = rogue_scorer.compute(generated_summaries, reference_summaries_lists)
    rouge_precision_result = rogue_scorer.compute(generated_summaries, reference_summaries_lists, score_type="precision")
    rouge_recall_result = rogue_scorer.compute(generated_summaries, reference_summaries_lists, score_type="recall")

    print("ROUGE results:")
    print(" - F1:")
    print("   - ROUGE-1: ", rouge_f1_result['rouge1'])
    print("   - ROUGE-2: ", rouge_f1_result['rouge2'])
    print("   - ROUGE-L: ", rouge_f1_result['rougeL'])
    print("   - ROUGE-Lsum: ", rouge_f1_result['rougeLsum'])
    print(" - precision:")
    print("   - ROUGE-1: ", rouge_precision_result['rouge1'])
    print("   - ROUGE-2: ", rouge_precision_result['rouge2'])
    print("   - ROUGE-L: ", rouge_precision_result['rougeL'])
    print("   - ROUGE-Lsum: ", rouge_precision_result['rougeLsum'])
    print(" - recall:")
    print("   - ROUGE-1: ", rouge_recall_result['rouge1'])
    print("   - ROUGE-2: ", rouge_recall_result['rouge2'])
    print("   - ROUGE-L: ", rouge_recall_result['rougeL'])
    print("   - ROUGE-Lsum: ", rouge_recall_result['rougeLsum'])

    sbleu_scorer = sacreBLEU()
    bleu_result = sbleu_scorer.compute(generated_summaries, reference_summaries_lists)

    print("BLEU results:")
    print(" - BLEU: ", bleu_result['score'])

    print("BLEURT results:")
    bleurt_scorer = score.BleurtScorer()
    bleurt_score = bleurt_scorer.score(references=reference_summaries, candidates=generated_summaries)
    bleurt_score = statistics.fmean(bleurt_score)
    print(" - BLEURT score: ", bleurt_score)

    bert_scorer = BERTScore(lang="en")
    bert_score = bert_scorer.compute(generated_summaries, reference_summaries_lists)

    print("BERTScore results:")
    print(" - BERT score: ", bert_score['f1'].mean())

    bart_scorer = BARTScore(model_name_or_path='facebook/bart-large-cnn')
    bart_score = bart_scorer.compute(generated_summaries, reference_summaries, agg="mean", batch_size=4)

    print("BARTScore results:")
    print(" - BART score: ", bart_score['score'].mean())

    print("All Results:")
    print(" - ROUGE-1 F1: ", rouge_f1_result['rouge1'])
    print(" - ROUGE-2 F1: ", rouge_f1_result['rouge2'])
    print(" - ROUGE-L F1: ", rouge_f1_result['rougeL'])
    print(" - ROUGE-Lsum F1: ", rouge_f1_result['rougeLsum'])
    print(" - ROUGE-1 precision: ", rouge_precision_result['rouge1'])
    print(" - ROUGE-2 precision: ", rouge_precision_result['rouge2'])
    print(" - ROUGE-L precision: ", rouge_precision_result['rougeL'])
    print(" - ROUGE-Lsum precision: ", rouge_precision_result['rougeLsum'])
    print(" - ROUGE-1 recall: ", rouge_recall_result['rouge1'])
    print(" - ROUGE-2 recall: ", rouge_recall_result['rouge2'])
    print(" - ROUGE-L recall: ", rouge_recall_result['rougeL'])
    print(" - ROUGE-Lsum recall: ", rouge_recall_result['rougeLsum'])
    print(" - BLEU: ", bleu_result['score'])
    print(" - BERT score: ", bert_score['f1'].mean())
    print(" - BART score: ", bart_score['score'].mean())
    print(" - BLEURT score: ", bleurt_score)

    metrics_json = {
        "episode_pids": episode_pids,
        "rouge_1_f1": str(rouge_f1_result['rouge1']),
        "rouge_2_f1": str(rouge_f1_result['rouge2']),
        "rouge_l_f1": str(rouge_f1_result['rougeL']),
        "rouge_lsum_f1": str(rouge_f1_result['rougeLsum']),
        "rouge_1_precision": str(rouge_precision_result['rouge1']),
        "rouge_2_precision": str(rouge_precision_result['rouge2']),
        "rouge_l_precision": str(rouge_precision_result['rougeL']),
        "rouge_lsum_precision": str(rouge_precision_result['rougeLsum']),
        "rouge_1_recall": str(rouge_recall_result['rouge1']),
        "rouge_2_recall": str(rouge_recall_result['rouge2']),
        "rouge_l_recall": str(rouge_recall_result['rougeL']),
        "rouge_lsum_recall": str(rouge_recall_result['rougeLsum']),
        "rouge_recall": str(rouge_recall_result['rouge1']),
        "bleu": str(bleu_result['score']),
        "bert_score": str(bert_score['f1'].mean()),
        "bart_score": str(bart_score['score'].mean()),
        "bleurt_score": str(bleurt_score)
      }

    return metrics_json


if __name__ == '__main__':
    main()
