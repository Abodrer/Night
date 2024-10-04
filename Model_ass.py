from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# تحميل النموذج والمفكك
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# إعداد النص
input_text = "الشخص الأول: مرحباً! كيف حالك اليوم؟\nالشخص الثاني:"

# ترميز النص
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# توليد النص
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# فك تشفير النص الناتج
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)