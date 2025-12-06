import torch
import json
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import os

# Increase the recursion limit
sys.setrecursionlimit(15000)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def train_single_piece(model, tokenizer, piece, training_args):
    # Tokenize the text
    tokenized_output = tokenizer(
        piece,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    # Set labels to be the same as input_ids
    tokenized_output["labels"] = tokenized_output["input_ids"].clone()

    # Create a dataset with a single sample
    class SingleSampleDataset(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            return {key: val.squeeze(0) for key, val in tokenized_output.items()}
        def __len__(self):
            return 1

    dataset = SingleSampleDataset()

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model on the current piece
    try:
        trainer.train()
    except RecursionError as e:
        print(f"Recursion error while training on piece: {piece[:50]}...\nError: {e}")

def train_batch(model, tokenizer, data_batch, training_args, batch_idx, num_batches):
    for i, piece in enumerate(data_batch):
        print(f"Training on piece {i + 1}/{len(data_batch)} in batch {batch_idx + 1}/{num_batches}: {piece[:50]}...")  # Print the first 50 characters of the piece
        train_single_piece(model, tokenizer, piece, training_args)

    # Save the model and tokenizer after processing each batch
    model_save_path = f"./saved_model_batch_{batch_idx + 1}"
    os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

def train():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token to be the same as the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Resize token embeddings to account for the new pad token
    model.resize_token_embeddings(len(tokenizer))

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the dataset from the file
    data_path = "data/raw/wikipedia-en-0.json"
    data = load_data(data_path)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=1,
        logging_steps=500,
        # Use GPU if available
        fp16=torch.cuda.is_available(),  # Enable 16-bit precision if using GPU
    )

    # Determine batch size
    batch_size = 2500
    num_batches = (len(data) + batch_size - 1) // batch_size

    # Train in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        print(f"Processing batch {batch_idx + 1}/{num_batches} (records {start_idx + 1}-{end_idx})")
        train_batch(model, tokenizer, data[start_idx:end_idx], training_args, batch_idx, num_batches)

if __name__ == "__main__":
    train()
