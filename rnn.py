# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:12:30 2019

@author: George Barker and Andre Zeromski
@author: watso
"""

import csv
import numpy as np
import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import sys


import matplotlib.pyplot as plt


class RNN:
    
    def __init__(self, word_dim, hidden_dim, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        
#def importData(filename):
#    # load text
#    file = open(filename, "rt")
#    text = file.read()
#    file.close()
#    # split into words by white space
#    words = text.split()
#    # convert to lower case
#    words = [word.lower() for word in words]
#    # remove punctuation from each word
#    import string
#    table = str.maketrans('', '', string.punctuation)
#    stripped = [w.translate(table) for w in words]
#    print(stripped[:1000])
    
def importData(filename):
    # load data
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    print(words[:100])
    
def main():
    importData("rnnDataset.csv")
    vocabulary_size = 8000
    unknown_token = "UNK"
    sentence_start_token = "START"
    sentence_end_token = "END"
    

if __name__ == "__main__":
    main()