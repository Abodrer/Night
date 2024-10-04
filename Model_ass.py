import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# تحميل البيانات من ملف JSON
with open('texts.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

texts = data.get('texts', [])

# إعداد الـ tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# ترميز النصوص
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

# إنشاء مجموعة بيانات
train_dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
})

# إعداد نموذج GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2')

# إعداد معلمات التدريب
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# إنشاء مدرب
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# بدء التدريب
trainer.train()

# وظيفة لتوليد نصوص
def generate_long_story(seed_text, total_words_count=50):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + total_words_count,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# استخدام النموذج لتوليد نص
long_story = generate_long_story("Once upon a time,", total_words_count=50)
print(long_story)