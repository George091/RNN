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

import re
from collections import Counter 
from nltk.corpus import stopwords
import string
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import WhitespaceTokenizer


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
    
def wordToVec(sentences):
    """ Given sentences as an array returns a Word2Vec model """
    # train model
    model = Word2Vec(sentences, min_count=1, iter = 5)
    # show vocab
    words = list(model.wv.vocab)
    print(words)
    print(model['sentence'])
    print(model)
    return words

def tokenizeSentences(text, vocabulary):
    """ Takes in a string of text and vocabulary and returns tokenized sentences """
    tokens = sent_tokenize(text)
    i = 0
    for sentence in tokens:
        tokens[i] = tokenizeWords(sentence)
        w = 0
        for word in tokens[i]:
            if word not in vocabulary:
                tokens[i][w] = "UNK"
            w += 1
        tokens[i].insert(0,"START")
        tokens[i].append("END")
        i += 1
    return tokens

def tokenizeWords(text):
    """ Takes in a string of text and returns an array of tokenized words """
    # replace all periods with a space
    text=re.sub("[.]", " ", text)
    # removes all punctuation except apostrophe
    text = re.sub("[^\w\d'\s]+",'',text)
    # split into words based on whitespace
    tokens = WhitespaceTokenizer().tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    return words
    
def findVocab(listOfWords):
    """ Take in a tokenized array of words and returns a list of top 8000 words """
    counterWords = Counter(listOfWords)
    most_occur = counterWords.most_common(8000)
    vocab = []
    for position in most_occur:
        vocab.append(position[0])
    return vocab
    
def importData(filename):
    """ Return the text from the file """
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    return text

def main():
#    vocabulary_size = 8000
#    unknown_token = "UNK"
#    sentence_start_token = "START"
#    sentence_end_token = "END"
    
    text = importData("rnnDataset.csv")
    tokenizedWords = tokenizeWords(text)
    vocabulary = findVocab(tokenizedWords)
    print(len(vocabulary))
    tokenizedSentences = tokenizeSentences(text, vocabulary)
    print(tokenizedSentences[:100])
    embedding = wordToVec(tokenizedSentences)
    print(set(embedding) ^ set(vocabulary))

if __name__ == "__main__":
    main()