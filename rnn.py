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
from gensim.models import Word2Vec
import sys

from collections import Counter 
from nltk.corpus import stopwords
import string
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import regexp_tokenize



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
    
def tokenizeWords(text):
    """ Takes in string of text and returns tokenized words """
    # split into words
    tokens = regexp_tokenize(text, "[\w']+")
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    print(tokens[:100])
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    print(words[:100])
    return words
    
def reduceVocab(listOfWords):
    """ Take in array of all words and returns a list of top 8000 words """
    counterWords = Counter(listOfWords)
    most_occur = counterWords.most_common(80)
    print(most_occur)
        
    
def wordToVec(sentences):
    """ Create Word2Vec embedding based on sentences """
    pass
    
def importData(filename):
    """ Return the text from the file """
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    return text

def main():
    text = importData("rnnDataset.csv")
    tokenizedWords = tokenizeWords(text)
    reduceVocab(tokenizedWords)
    
    vocabulary_size = 8000
    unknown_token = "UNK"
    sentence_start_token = "START"
    sentence_end_token = "END"
    

if __name__ == "__main__":
    main()