#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:03:08 2019

@author: georgebarker andrezeromski
"""

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Bidirectional, LSTM, Dropout
import pickle

# constants
top_words = 5000
max_review_length = 600
embedding_vector_length = 5

# Load Data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

# Pad and reduce length of input
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# Create model
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## Load RNN from pickle
pickle_in = open("RNN-DEEP-final","rb")
model = pickle.load(pickle_in)

# Summary of model
print(model.summary())

# Train model
#model.fit(x_train, y_train, epochs=4, batch_size=64)

# Evaluate model
predictions = model.evaluate(x_test, y_test)
print("accuracy: %.2f%%" % (predictions[1]*100))

#pickle_out = open("RNN-DEEP-final-1","wb")
#pickle.dump(model, pickle_out)
#pickle_out.close()
