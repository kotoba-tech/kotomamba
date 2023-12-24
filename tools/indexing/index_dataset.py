from pathlib import Path
import os

import argparse
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("kotoba-recipes")
    parser.add_argument("--data-file-path", type=str, default="")
    args = parser.parse_args()
    return args


def create_index_file(data_file_path: str) -> None:
    """create index file for high speed random access

    Args:
        data_file_path (str): ex: datasets/instruction_data.jsonl
    """

    def is_valid_utf8(line: str) -> bool:
        try:
            line.encode("utf-8").decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    dataset_dir = Path(data_file_path).parent
    index_cache_dir = dataset_dir / ".index_cache"
    os.makedirs(index_cache_dir, exist_ok=True)
    index_file_path = index_cache_dir / str(os.path.basename(data_file_path)).replace(".jsonl", ".idx")

    print(f"Creating index file: {index_file_path}", flush=True)
    # create index file(tmp)
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", dir=index_cache_dir) as temp_file:
        with open(data_file_path, "r", encoding="utf-8") as data_file:
            while True:
                offset = data_file.tell()
                line = data_file.readline()
                if not line:
                    break
                if line.endswith("\n") and is_valid_utf8(line):
                    temp_file.write(f"{offset}\n")
                else:
                    print(f"invalid: {line}", flush=True)
        # rename temp file to index file
        os.rename(temp_file.name, index_file_path)

    print(f"indexing Done: {index_file_path}", flush=True)


if __name__ == "__main__":
    args = parse_args()

    # indexing
    if args.data_file_path:
        create_index_file(args.data_file_path)
    else:
        raise ValueError("data_file_path is required")
