# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import preprocessing
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

data_csv = pd.read_csv(os.path.join(os.getcwd(), "rotten_tomatoes_reviews.csv"), skipinitialspace=True)

reviews = data_csv["Review"]
target = data_csv["Freshness"]

# preprocessing, take from F. Chollet's Deep Learning with Python
data = [reviews[i] for i in range(len(reviews))]
data = np.array(data)
# max length of each review
maxlen = 100
# max words for the tokenizer (and for the embedding layer later)
max_words = 30000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
# tokenised sequences
sequences = tokenizer.texts_to_sequences(data)

# padding
seq_train = preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)

# train and test datasets
x_train, x_test, y_train, y_test = train_test_split(seq_train, target)

model = Sequential()
model.add(Embedding(max_words, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# plotting loss
plt.figure()
plt.plot(history.history['loss'])
plt.grid()

# predictions
accs = model.evaluate(x_test, y_test)

# accuracy:
print("Accuracy = ", accs)
