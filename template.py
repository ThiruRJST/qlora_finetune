import logging
import os

from pathlib import Path

logging.basicConfig(level=logging.INFO)

project_name = "qlora_finetune_phi3"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/configs/__init__.py",
    f"src/{project_name}/configs/configuration.py",
    f"src/{project_name}/trainer/__init__.py",
    f"src/{project_name}/trainer/training_engine.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/data_sampler.py",
    f"src/{project_name}/main.py",
    f"configs/config.yaml",
    "README.md",
    "requirements.txt",
    "setup.py",
    "Dockerfile",
    "research/research.ipynb"
    
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        if not os.path.exists(filedir):
            logging.info(f"Creating directory: {filedir}")
            os.makedirs(filedir, exist_ok=True)
        else:
            logging.info(f"Directory already exists: {filedir}")
    
    
    if not os.path.exists(filepath):
        logging.info(f"Creating file: {filepath}")
        with open(filepath, "w") as f:
            pass
    else:
        logging.info(f"File already exists: {filepath}")