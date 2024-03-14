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

