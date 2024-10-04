import json
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

# تحميل النموذج والمفكك
tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-3B')
model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-3B')

def load_conversations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get("conversations", [])

def generate_response(input_text):
    try:
        # إعداد النص
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        # توليد الرد مع تحسينات
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )

        # فك تشفير النص الناتج
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    except Exception as e:
        return f"An error occurred: {e}"

# واجهة المستخدم البسيطة
if __name__ == "__main__":
    print("Welcome to the BlenderBot conversation!")
    
    # تحميل المحادثات من ملف JSON
    conversations = load_conversations('texts.json')
    
    for input_text in conversations:
        print("You (from JSON): ", input_text)
        response = generate_response(input_text)
        print("BlenderBot: ", response)

    print("\nNow you can have a conversation. Type 'exit' or 'quit' to stop.")
    
    # تفاعل مع المستخدم
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = generate_response(user_input)
        print("BlenderBot: ", response)