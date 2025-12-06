from ray import tune
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

def train_model(config):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    trainer.train()

tune.run(
    train_model,
    config={
        "num_train_epochs": tune.grid_search([1, 3]),
        "per_device_train_batch_size": tune.grid_search([2, 4]),
    },
)
