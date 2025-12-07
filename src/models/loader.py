from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
