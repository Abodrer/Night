from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

# Set the model to evaluation mode
model.eval()

# Prepare the input text
input_text = "hi dude how are ya?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Create attention mask
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

# Generate the text with improved parameters
with torch.no_grad():
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,   # Control randomness
        top_k=50,          # Limit the number of highest probability vocabulary
        top_p=0.95,        # Nucleus sampling
        do_sample=True     # Enable sampling
    )

# Decode the generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)