import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
import re
import string

def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""  # Return empty string for non-text elements or empty plot summaries

    # Convert text to lowercase
    text = text.lower()
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Step 1: Load and preprocess the data
data = pd.read_csv('test.txt', delimiter='\t', header=None)
plot_summaries = data[1].tolist()  # Get the plot summaries from the second column

print("Number of plot summaries before cleaning:", len(plot_summaries))

# Step 2: Clean the data and remove non-text elements
plot_summaries_cleaned = [clean_text(summary) for summary in plot_summaries]
plot_summaries_cleaned = [text for text in plot_summaries_cleaned if text]

print(plot_summaries_cleaned)
print("Number of plot summaries after cleaning:", len(plot_summaries_cleaned))

# Rest of the code...

# Step 3: Tokenization and Vocabulary Building
max_words = 10000  # Set the maximum number of words in the vocabulary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(plot_summaries_cleaned)
total_words = len(tokenizer.word_index) + 1

# Step 4: Sequencing the Data
sequences = tokenizer.texts_to_sequences(plot_summaries_cleaned)
input_sequences = pad_sequences(sequences, padding='pre', truncating='pre', maxlen=None)  # Pad all sequences to the same length

# Step 5: Creating input and output sequences
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Step 6: Training the Model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=128)  # Use a larger batch size

# Step 7: Text Generation
seed_text = "In a world full of"
next_words = 100
temperature = 0.5  # Adjust the temperature value

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=X.shape[1], padding='pre')
    predictions = model.predict(token_list, verbose=0)[0]
    
    # Adjust the predictions using temperature
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    
    # Sample the next word using the probability distribution
    predicted_index = np.random.choice(len(predictions), p=predictions)
    
    predicted_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break
    seed_text += " " + predicted_word

print(seed_text)
