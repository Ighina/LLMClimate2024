import json
import os
from urllib.request import urlopen

from datasets import load_dataset
import pandas as pd

import json
import os
from urllib.request import urlopen

import pandas as pd

def download_jsonl_file(url: str, out_file: str):
  """ Write locally a remote jsonl file
  Parameters
  ----------
  url:      str
            the path to the remote jsonl file
  out_file: str
            the path to the local file where to store the jsonl file
  Returns
  ----------
  None
  The function writes the remote jsonl file in the location specified at out_file
  """
  with urlopen(url) as f_in, open(out_file, "w") as f_out:
      data_file = [json.loads(x) for x in f_in.readlines()]
      for line in data_file:
        json.dump(line, f_out)
        f_out.write("\n")

def change4web(url):
  """ Change special characters for getting valid url for downloading
  Parameters
  ----------
  url:  str
        The url to download
  Returns
  ----------
  url:  str
        The changed url, to be used when downloading
  """
  return url.replace(" ", "%20").replace(r":", "%3A").replace(",","%2C")

# one of ['climate_stance', 'climate_eng', 'climate_fever', 'climatext', 'clima_insurance', 'clima_insurance_plus', 'clima_cdp', 'clima_qa', 'scidcc']
def get_climabench_subset(subset: str="climate_stance",
                          delete_local: bool=False):
  """Download a subset of the climabench benchmark
  Parameters
  ----------
  subset:       str
                which subset, by name (see comment above for available options)
  delete_local: bool
                whether to delete the files downloaded locally after extraction
  Return
  ----------
  dataset:      datasets.Dataset
                an instance of the Dataset class containing the required subset
                splitted in training, validation and test set
  """
  remote_dir = "https://huggingface.co/datasets/iceberg-nlp/climabench/resolve/main/"
  try:
    #########################################################################
    # CLIMATE FEVER DATASET
    #########################################################################
    if subset == "climate_fever":
      local_dir = os.path.join("all_data", "ClimateFEVER", "test-data")
      # create directory locally
      os.makedirs(local_dir)
      local_dir = os.path.join(local_dir, "climate-fever-dataset-r1.jsonl")

      # define file to download
      url = os.path.join(remote_dir, local_dir)
      download_jsonl_file(url, local_dir)

    #########################################################################
    # CLIMATEXT DATASET
    #########################################################################

    elif subset == "climatext":
      local_dir = os.path.join("all_data", "ClimaText")

      # create directory locally
      os.makedirs(os.path.join(local_dir, "train-data"))
      os.makedirs(os.path.join(local_dir, "test-data"))
      os.makedirs(os.path.join(local_dir, "dev-data"))

      # download files locally
      #https://huggingface.co/datasets/iceberg-nlp/climabench/resolve/main/all_data/ClimaText/train-data/AL-10Ks.tsv%20%3A%203000%20(58%20positives%2C%202942%20negatives)%20(TSV%2C%20127138%20KB).tsv?download=true
      files = [
                r"train-data/AL-10Ks.tsv : 3000 (58 positives, 2942 negatives) (TSV, 127138 KB).tsv",
                r"train-data/AL-Wiki (train).tsv",
                r"dev-data/Wikipedia (dev).tsv",
                r"test-data/Claims (test).tsv",
                r"test-data/Wikipedia (test).tsv",
                r"test-data/10-Ks (2018, test).tsv",
                ]
      for fl in files:
        # change whitespaces colons and commas for downloading the url
        fl_web = fl.replace(" ", "%20").replace(r":", "%3A").replace(",","%2C")
        pd.read_csv(os.path.join(remote_dir, local_dir, fl_web), sep='\t').to_csv(os.path.join(local_dir, fl), sep='\t')

    #########################################################################
    # CLIMA QA DATASET (maybe too big for us)
    #########################################################################

    elif subset == "clima_qa":
      local_dir = os.path.join("all_data", "CDP")
      categories = ["Cities/Cities Responses", "States", "Corporations/Corporations Responses/Climate Change"]
      # create directory locally
      for category in categories:
        os.makedirs(os.path.join(local_dir, category))

        fl_train = os.path.join(category, "train_qa.csv")
        fl_val = os.path.join(category, "val_qa.csv")
        fl_test = os.path.join(category, "test_qa.csv")

        # download files locally
        pd.read_csv(os.path.join(remote_dir, local_dir, change4web(fl_train))).to_csv(os.path.join(local_dir, fl_train))
        pd.read_csv(os.path.join(remote_dir, local_dir, change4web(fl_val))).to_csv(os.path.join(local_dir, fl_val))
        pd.read_csv(os.path.join(remote_dir, local_dir, change4web(fl_test))).to_csv(os.path.join(local_dir, fl_test))

    #########################################################################
    # SCIDCC DATASET
    #########################################################################

    elif subset == "scidcc":
      local_dir = os.path.join("all_data", "SciDCC")
      os.makedirs(local_dir)
      pd.read_csv(os.path.join(remote_dir, local_dir, "SciDCC.csv")).to_csv(os.path.join(local_dir, "SciDCC.csv"))

    #########################################################################
    # CLIMATE STANCE, CLIMATE ENG and CLIMATE INSURANCE DATASETS
    #########################################################################

    else:
      if subset=="clima_cdp":
        local_dir = os.path.join("all_data", "CDP", "Cities", "Cities Responses")
      else:
        # convert name of subset to correct format
        local_dir=os.path.join("all_data", "".join([r[0].upper()+r[1:] for r in subset.split("_")]))

      # create directory locally
      os.makedirs(local_dir)

      # download files locally
      pd.read_csv(os.path.join(remote_dir, change4web(local_dir),"train.csv")).to_csv(os.path.join(local_dir, "train.csv"))
      pd.read_csv(os.path.join(remote_dir, change4web(local_dir),"val.csv")).to_csv(os.path.join(local_dir, "val.csv"))
      pd.read_csv(os.path.join(remote_dir, change4web(local_dir),"test.csv")).to_csv(os.path.join(local_dir, "test.csv"))

    # create dataset
  except FileExistsError:
    pass
  dataset  = load_dataset("iceberg-nlp/climabench", subset, trust_remote_code = True)

  if delete_local:
    os.rmdir(local_dir)
  return dataset
