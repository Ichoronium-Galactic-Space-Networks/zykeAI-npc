import math

from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def evaluate(model_path, tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results_eval",
        per_device_eval_batch_size=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss")
    ppl = math.exp(eval_loss) if eval_loss is not None else float("nan")
    print(f"Validation loss: {eval_loss:.4f}")
    print(f"Validation perplexity: {ppl:.2f}")


if __name__ == "__main__":
    evaluate("./results", "./results")
