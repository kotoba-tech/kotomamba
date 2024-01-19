import sys
import os

current_path: str = os.getcwd()
sys.path.append(f"{current_path}/src")

from llama_recipes.finetuning import main

if __name__ == "__main__":
    main()
