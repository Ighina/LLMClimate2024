# LLMClimate2024
Code for our paper at NLPClimate2024

## Create Virtual Environment
If not using Docker, create and activate a virtual environment to install all the dependencies.
Do that with conda as following:
```
conda create -n climatellm python=3.10.12
conda activate climatellm
```

Or with venv as following:
```
python -m venv climatellm
source climatellm/bin/activate
python -m pip install python==3.10.12
```

## Install Dependencies
To install all the dependencies, be sure to have poetry installed and to be inside your virtual environment. Then, from inside this repo, use poetry as follow
```
poetry install
```
All dependencies should be automatically installed inside your virtual environment.

## Obtain UniEval
As a start, clone the UniEval repository for evaluation.
```
git clone https://github.com/maszhongming/UniEval.git
```

If not using Docker, install UniEval packages as follow:
```
cd UniEval
pip install requirements.txt
cd ..
```

## Use the evaluation script
Now you can run the main programme as following:
```
python summary_evaluation.py --model <model_name> --device <gpu/cpu> --output <output_directory> -sump
```
With  the above parameters being:
--model: the name of the LLM to use. It needs to be among the ones available from Huggingface Hub

--device: whether to use cpu or gpu

--output: where to store results locally

-sump: this option makes so that if input longer than maximum model inputs are found, we iteratively summarise those and then summarise the concatenation of these partial results. This is currently the only mechanism we have to overcome token limit problems and, as such, you should always include this option.

## Use with Docker
The evaluation script can also be used with Docker, via the building file provided in the repository. Once you have a properly installed [Docker](https://www.docker.com/get-started/), be sure to have admin rights and then do one of two things:

1) If you have a GPU and you want to use it, run:
```
docker compose run evaluate_gpu --model <model_name> --output <output_directory> -sump
```

2) If you do not have a GPU or you want to run on CPU:
```
docker compose run evaluate_cpu --model <model_name> --output <output_directory> -sump
```

Where the various arguments are described in more details in the previous section.

If you do not have admin rights from where you are running the script, you might get a permission error. If you are running the script from a Linux machine, be sure to add sudo before the command as following:
```
sudo docker compose run evaluate_gpu --model <model_name> --output <output_directory> -sump
```

## Weights & Biases integration
The code supports [Weights & Biases](https://wandb.ai/site) to log results. If you have a profile on Weights & Biases you can use the integration by adding your username and token API private key into the configuration file named [config.yaml](config/config.yaml) inside the config folder. Just fill the details in the file as following:
- entity: your username on Weights and Biases
- project: the Weights and Biases project where to log the results
- key: your token API private key

