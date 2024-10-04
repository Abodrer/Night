from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model to evaluation mode
model.eval()

# Prepare the input text
input_text = "Person 1: Hi! How are you today?\nPerson 2:"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Create attention mask
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

# Generate the text
with torch.no_grad():
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1)

# Decode the generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)