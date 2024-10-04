import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load texts from a JSON file
with open('texts.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

texts = data.get('texts', [])  # Using get for safety

# Preparing the data (if needed)
texts = [text.strip() for text in texts if text.strip()]

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to generate longer text
def generate_long_story(seed_text, total_words_count=50):
    # Encode the seed text
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')

    # Generate text
    output = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + total_words_count,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Prevent repetition
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# Using the model to generate a long text
long_story = generate_long_story("Once upon a time,", total_words_count=50)
print(long_story)