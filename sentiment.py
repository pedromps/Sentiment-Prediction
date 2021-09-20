# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from keras import preprocessing
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.preprocessing.text import Tokenizer



data_csv = pd.read_csv(os.path.join(os.getcwd(), "rotten_tomatoes_reviews.csv"), skipinitialspace=True)

reviews = data_csv["Review"]
target = data_csv["Freshness"]

# preprocessing

data = [reviews[i] for i in range(len(reviews))]
data = np.array(data)

texts = data
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 30000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

seq_train = preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_words, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(seq_train, target, epochs=10, batch_size=128, validation_split=0.2)
