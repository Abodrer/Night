import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, Input, Attention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Load texts from a JSON file
with open('texts.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

texts = data.get('texts', [])  # Using get for safety

# Preparing the data
texts = [text.strip() for text in texts if text.strip()]

# Creating input and target sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Padding sequences
max_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='pre')

# Separating inputs and targets
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = keras.utils.to_categorical(y, num_classes=total_words)

# Building the model
inputs = Input(shape=(max_length - 1,))
embedding_layer = Embedding(total_words, 100)(inputs)
bidirectional_lstm = Bidirectional(LSTM(150, return_sequences=True))(embedding_layer)
bidirectional_lstm = Dropout(0.3)(bidirectional_lstm)
bidirectional_lstm = Bidirectional(LSTM(150))(bidirectional_lstm)  # Adding another LSTM layer

# Adding Attention layer
attention = Attention()([bidirectional_lstm, bidirectional_lstm])
attention = Flatten()(attention)
attention = Dropout(0.3)(attention)

# Final layer
outputs = Dense(total_words, activation='softmax')(attention)

# Compiling the model
model = Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model with callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X, y, epochs=100, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Function to generate longer text with improved control over repetition
def generate_long_story(seed_text, total_words_count=50):
    story = seed_text
    used_words = set()
    for _ in range(total_words_count):
        token_list = tokenizer.texts_to_sequences([story])[0]
        token_list = pad_sequences([token_list], maxlen=max_length - 1, padding='pre')
        
        predicted = model.predict(token_list, verbose=0)
        predicted_probs = predicted[0]

        # Applying penalty for repeated words
        for word in used_words:
            if word in tokenizer.word_index:
                idx = tokenizer.word_index[word]
                predicted_probs[idx] *= 0.5

        # Top-k sampling instead of argmax
        top_k = 5
        top_indices = np.argsort(predicted_probs)[-top_k:]  # Get top-k indices
        chosen_index = np.random.choice(top_indices)  # Randomly select from top-k
        output_word = tokenizer.index_word[chosen_index]
        
        if output_word:
            used_words.add(output_word)
            story += " " + output_word
    
    return story

# Using the model to generate a long text
long_story = generate_long_story("Once upon a time,", total_words_count=50)
print(long_story)