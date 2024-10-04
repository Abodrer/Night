import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, Input, Attention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json

# Load texts from a JSON file
with open('texts.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    texts = data['texts']

# Preparing the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

# Creating input and target sequences
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

# Custom Loss Function incorporating regression-like aspects
def custom_loss(y_true, y_pred):
    # Using a combination of cross-entropy and a regularization term
    cross_entropy_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # Regularization term (L2 Regularization as an example)
    reg_loss = 0.01 * tf.reduce_sum([tf.nn.l2_loss(v) for v in model.trainable_variables])
    
    return cross_entropy_loss + reg_loss

# Building the model with complex enhancements
inputs = Input(shape=(max_length - 1,))
embedding_layer = Embedding(total_words, 100)(inputs)
bidirectional_lstm = Bidirectional(LSTM(150, return_sequences=True))(embedding_layer)
bidirectional_lstm = Dropout(0.3)(bidirectional_lstm)

# Enhanced Attention Layer
attention = Attention()([bidirectional_lstm, bidirectional_lstm])
attention = Flatten()(attention)
attention = Dropout(0.3)(attention)

# Final layer
outputs = Dense(total_words, activation='softmax')(attention)

# Compiling the model with the custom loss function
model = Model(inputs, outputs)
model.compile(loss=custom_loss, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Training the model
model.fit(X, y, epochs=200, verbose=1)

# Function to generate longer text with improved control over repetition
def generate_long_story(seed_text, total_words_count=50):
    story = seed_text
    used_words = set()
    for _ in range(total_words_count):
        token_list = tokenizer.texts_to_sequences([story])[0]
        token_list = pad_sequences([token_list], maxlen=max_length - 1, padding='pre')
        
        predicted = model.predict(token_list, verbose=0)
        predicted_probs = predicted[0]
        
        # Applying penalty for repeated words and using advanced probability distribution
        for word in used_words:
            if word in tokenizer.word_index:
                idx = tokenizer.word_index[word]
                predicted_probs[idx] *= 0.5  # Reduce the probability of choosing repeated words

        # Introduce a randomness factor
        predicted_probs = predicted_probs ** (1 / 1.5)  # Sharpen the probability distribution
        predicted_probs /= np.sum(predicted_probs)  # Normalize

        predicted_word_index = np.random.choice(range(len(predicted_probs)), p=predicted_probs)
        output_word = tokenizer.index_word[predicted_word_index]

        used_words.add(output_word)
        story += " " + output_word

    return story.strip()

# Using the model to generate a long text
long_story = generate_long_story("Once upon a time,", total_words_count=50)
print(long_story)