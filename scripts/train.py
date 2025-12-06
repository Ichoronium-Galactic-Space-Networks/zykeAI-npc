import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def train():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Use EOS as padding to avoid padding errors and expand embeddings to include it
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load JSON array of strings as text dataset
    raw_datasets = load_dataset(
        "json", data_files="data/raw/wikipedia-en-0.json", split="train"
    )

    # Create a small validation slice to monitor training
    split_datasets = raw_datasets.train_test_split(test_size=0.01, seed=42)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
        )

    tokenized_datasets = split_datasets.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=500,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()

    model_save_path = "./saved_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")


if __name__ == "__main__":
    train()
