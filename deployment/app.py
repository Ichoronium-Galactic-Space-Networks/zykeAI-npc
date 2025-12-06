from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pretrained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Set the pad_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

# Input text or prompt
input_text = "Was Hamlet married?"

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True, max_length=100, truncation=True)

# Manually set the attention mask
attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

# Generate output
output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)

# Decode and display output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)
