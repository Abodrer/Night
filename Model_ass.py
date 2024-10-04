from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

# تحميل النموذج والمفكك
tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-3B')
model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-3B')

# إعداد النص
input_text = "Hi! How are you today? Do you like anime? "
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# توليد الرد
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)

# فك تشفير النص الناتج
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)