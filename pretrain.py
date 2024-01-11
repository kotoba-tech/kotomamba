import sys
import os

current_path: str = os.getcwd()
sys.path.append(f"{current_path}/src")

import fire
from llama_recipes.finetuning import main

if __name__ == "__main__":
    fire.Fire(main)
