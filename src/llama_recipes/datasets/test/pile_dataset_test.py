# python -m llama_recipes.datasets.test.pile_dataset_test
import os, time
import random
import torch 
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer
from llama_recipes.datasets import get_pile_dataset 
from IPython import embed

@dataclass
class DataConfig:
    train_data_path: str
    val_data_path: str
    context_size: int

def load_next_encoded_tokens(sequence_pointer, sequence_length, bin_buffer, dtype):
    encoded_text: list[int] = np.frombuffer(
        bin_buffer,
        dtype=dtype,
        count=sequence_length,
        offset=sequence_pointer,
    )
    return encoded_text

def print_batch(dataset, ds_index = 0):
    '''
    Visualize all examples in the dataset
    '''
    output = dataset.__getitem__(ds_index)
    print("="*50 + "input" + "="*50 )
    print(tokenizer.decode(output["input_ids"]))
    print("="*100)
    print("output: {}".format(output["labels"]))
    print("attention_mask: {}".format(output["attention_mask"]))

def test_train_batch(dataset, ds_index = 0):
    '''
    Check the batch of training
    '''
    # Retrieve the item from the dataset using the given index
    output = dataset.__getitem__(ds_index)
    current_index = 0
    total_offset_tokens = dataset.max_words * ds_index
    current_number_of_tokens = 0
    batch_processing_started = False
    batch_input_ids = []
    num_docs = len(dataset.index)

    # Iterate through the dataset
    while current_index < num_docs:
        # Extract encoded text information
        sequence_pointer, sequence_length, sequence_mode = dataset.index[current_index]
        encoded_text = load_next_encoded_tokens(sequence_pointer, sequence_length, dataset.bin_buffer, dataset.index.dtype)
        
        # Start batch processing if the current number of tokens exceeds the offset
        if not batch_processing_started and (current_number_of_tokens + len(encoded_text) > total_offset_tokens):
            batch_processing_started = True
            current_offset = total_offset_tokens - current_number_of_tokens
            batch_input_ids.extend(encoded_text[current_offset:])
        elif batch_processing_started:
            batch_input_ids.extend(encoded_text)
        else:
            current_number_of_tokens += len(encoded_text) 
                
        # Break the loop if the batch size limit is reached or it's the last index
        if len(batch_input_ids) >= dataset.max_words or current_index == num_docs - 1:
            break

        current_index += 1

    # Convert the batch input IDs into a tensor and assert equality with the dataset output
    groundtruth_input = torch.tensor(np.array(batch_input_ids[:dataset.max_words]).astype(np.int32)) 
    # Prepare the groundtruth output tensor with appropriate padding and values
    groundtruth_output = torch.zeros(dataset.max_words, dtype=torch.long)
    groundtruth_output[-1] = -100
    groundtruth_output[:len(groundtruth_input) - 1] = groundtruth_input[1:]
    # Assertions to verify the consistency with the dataset output
    assert torch.all(groundtruth_input == output['input_ids'][output["attention_mask"].bool()]), \
        "Ground-truth input does not match dataset output"
    assert torch.sum(output["attention_mask"]) == len(groundtruth_input), \
        "Ground-truth attention mask does not match dataset output"
    assert torch.all(output["labels"] == groundtruth_output), \
        "Ground-truth output does not match dataset output"

def test_validation_batch(dataset, ds_index = 0):
    '''
    Check the batch of training
    '''
    # Retrieve the item from the dataset using the given index
    output = dataset.__getitem__(ds_index)
    sequence_pointer, sequence_length, sequence_mode = dataset.index[ds_index]
    encoded_text = load_next_encoded_tokens(sequence_pointer, sequence_length, dataset.bin_buffer, dataset.index.dtype).astype(np.int32)  # Convert to int32 to avoid overflow issues

    # Convert batch input IDs to a PyTorch tensor and truncate/pad to match dataset's max words
    groundtruth_input = torch.tensor(encoded_text[:dataset.max_words], dtype=torch.long)
    # Prepare the groundtruth output tensor with appropriate padding and values
    groundtruth_output = torch.zeros(dataset.max_words, dtype=torch.long)
    groundtruth_output[-1] = -100
    groundtruth_output[:len(groundtruth_input) - 1] = groundtruth_input[1:]
    # Assertions to verify the consistency with the dataset output
    assert torch.all(groundtruth_input == output['input_ids'][output["attention_mask"].bool()]), \
        "Ground-truth input does not match dataset output"
    assert torch.sum(output["attention_mask"]) == len(groundtruth_input), \
        "Ground-truth attention mask does not match dataset output"
    assert torch.all(output["labels"] == groundtruth_output), \
        "Ground-truth output does not match dataset output"


if __name__ == "__main__":
    # Configuration for the dataset
    context_size = 4096
    data_path = "/groups/gcd50698/fujii/datasets/pile/bin/pile-mamba-validation_text_document.bin"
    dataset_config = DataConfig(data_path, data_path, context_size)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)

    # Load the training dataset
    train_dataset = get_pile_dataset(dataset_config, tokenizer, "train")

    # Print random 10 examples from the training batch
    for ind in random.sample(range(len(train_dataset)), 10):
        print_batch(train_dataset, ind)
        time.sleep(15)
        os.system("clear")

    # Testing training batches
    for i in tqdm(range(len(train_dataset))):
        test_train_batch(train_dataset, i)
    print("=" * 100)
    print("Passing the tests for all the training batches")

    # Load the validation dataset and test validation batches
    validation_dataset = get_pile_dataset(dataset_config, tokenizer, "validation")
    for i in tqdm(range(len(validation_dataset))):
        test_validation_batch(validation_dataset, i)
    print("=" * 100)
    print("Passing the tests for all the validation batches")
