

#%%
from openai import OpenAI





# %%
import json
from openai import OpenAI

# Load API key from JSON file
with open("./api_key.json", "r") as file:
    api_keys = json.load(file)

# Initialize OpenAI client with the loaded API key
client = OpenAI(
    api_key=api_keys["api_openai"]
)

# Make a request to OpenAI's API
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
        {"role": "user", "content": "write a haiku about AI"}
    ]
)

# Print the response
print(completion.choices[0].message)






##############################################
##############################################

#%%
import os

os.environ["PROJECT_PATH"] = "/Users/wentaojiang/Documents/GitHub/ChartMimic"

print(os.environ["PROJECT_PATH"])  # Verify it's set

from dotenv import load_dotenv

load_dotenv()  # Load environment variables

print(os.environ.get("PROJECT_PATH"))  # Debugging: Check if it's loaded correctly


# %% Import required modules
import sys
import os
import re
import wandb
import warnings
import yaml
import json
import time
from dotenv import load_dotenv
from tasks import load_task

# Define available tasks
TASKS = [
    "chart2code",
    "gpt4evaluation",
    "code4evaluation",
    "autoevaluation",
    "chartedit",
]

# %% Load environment variables
load_dotenv(override=True, 
    dotenv_path='/Users/wentaojiang/Documents/GitHub/ChartMimic/.env')  # Take environment variables from .env (e.g., API keys, paths)

# %% Manually Define Arguments (Instead of CLI Parsing)
cfg_path = "../eval_configs/direct_generation.yaml"  # Set your configuration file path
tasks_to_run = ["chart2code"]  # List the tasks to execute
model_name = "gpt-4-vision-preview"  # Model choice
evaluation_dir = ""  # Set the evaluation directory

# %% Path Constructor for YAML
def path_constructor(loader, node):
    path_matcher = re.compile(r"\$\{([^}^{]+)\}")
    """ Extract the matched value, expand env variable, and replace the match """
    value = node.value
    match = path_matcher.match(value)
    if match:
        env_var = match.group()[2:-1]
        return os.environ.get(env_var, "") + value[match.end():]
    return value

# %% Load Configuration
def load_config(cfg_path):
    path_matcher = re.compile(r"\$\{([^}^{]+)\}")
    yaml.add_implicit_resolver("!path", path_matcher)
    yaml.add_constructor("!path", path_constructor)

    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config["llm"], config["agent"], config["run"]

# Load configurations
llm_config, agent_config, run_config = load_config(cfg_path)
llm_config = llm_config[model_name]
llm_config["model"] = model_name

# %% Check and Create Required Log Paths
def check_log_paths_are_ready(log_dir, baseline_dir):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "logs"), exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)

    all_results_path = os.path.join(log_dir, "all_results.txt")
    if not os.path.exists(all_results_path):
        with open(all_results_path, "w") as f:
            f.write("")
    
    return True

# %% Run Evaluation Tasks
def run_tasks(tasks, run_config, llm_config, agent_config, evaluation_dir):
    s = time.time()

    for task_name in tasks:
        if task_name in ["gpt4evaluation", "autoevaluation"]:
            run_config[task_name]["generated_dataset_dir"] = evaluation_dir
        
        if task_name not in TASKS:
            raise ValueError(f"Task {task_name} is not supported")
        
        task = load_task(task_name, run_config[task_name], llm_config, agent_config)
        task.run()

    print("Time taken: ", time.time() - s)

#%% Run the tasks
run_tasks(tasks_to_run, run_config, llm_config, agent_config, evaluation_dir)


# %%
import openai

models = openai.Client().models.list()
print([model.id for model in models.data])


# %%
