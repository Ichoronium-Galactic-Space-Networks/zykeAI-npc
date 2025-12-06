import os
from datasets import load_dataset

def prepare_data(input_file, output_dir):
    # Read raw data from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Split raw text into documents or samples
    documents = raw_text.split("\n")  # Adjust the delimiter if necessary

    # Preprocess and save the dataset
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.txt"), "w", encoding='utf-8') as f:
        for doc in documents:
            # Preprocess each document (e.g., tokenization, cleaning, etc.)
            # Here you can apply any specific preprocessing steps needed for your data
            processed_doc = doc.strip()  # Example: remove leading/trailing whitespace

            # Write preprocessed document to file
            if processed_doc:  # Skip empty documents
                f.write(processed_doc + "\n")  # Adjust formatting as needed

if __name__ == "__main__":
    input_file = "data/raw/data.txt"  # Path to your raw data file
    output_dir = "data/processed"  # Directory to save the processed data

    prepare_data(input_file, output_dir)

