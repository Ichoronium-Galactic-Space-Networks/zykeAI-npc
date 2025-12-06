from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(model_path, tokenizer_path, prompt, max_length=100):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_text = generate_text("./results", "./results", prompt)
    print(generated_text)
