# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:12:30 2019

@author: George Barker and Andre Zeromski
@author: watso
"""

import csv
import numpy as np

from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from collections import Counter 
from nltk.corpus import stopwords


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
    """ Given an array of tokenized words returns a Word2Vec model """
    model = Word2Vec(sentences, size = 10, min_count=1, iter = 5)
    return model

def tokenizeLines(arrayOfLines):
    """ Takes in a array of lines of text and returns tokenized lines """
    i = 0
    for line in arrayOfLines:
        arrayOfLines[i] = tokenizeText(line[0])
        i += 1
    flattened  = [val for sublist in arrayOfLines for val in sublist]
    vocabulary = findVocab(flattened)
    l = 0
    for line in arrayOfLines:
        w = 0
        for word in line:
            if word not in vocabulary:
                arrayOfLines[l][w] = "UNK"
            w += 1
        arrayOfLines[l].insert(0,"START")
        arrayOfLines[l].append("END")
        l += 1
    return arrayOfLines

def tokenizeText(text):
    """ Takes in a string of text and returns an array of tokenized words """
    # tokenize words based on seperating whitespace
    tokens = word_tokenize(text)
    # lowercase all words
    tokens = [token.lower() for token in tokens]
    # remove stop words
    stopWords = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stopWords]
    return tokens
    
def findVocab(listOfWords):
    """ Take in a tokenized array of words and returns a list of top 8000 words """
    counterWords = Counter(listOfWords)
    most_occur = counterWords.most_common(8000)
    vocab = []
    for position in most_occur:
        vocab.append(position[0])
    return vocab
    
def importData(filename):
    """ Return the text from the file where each line is an element in an array """
    f = open(filename)
    csv_f = csv.reader(f)
    text = []
    for row in csv_f:
        text.append(row)
    return text

def main():
#    vocabulary_size = 8000
#    unknown_token = "UNK"
#    sentence_start_token = "START"
#    sentence_end_token = "END"
    text = importData("rnnDataset.csv")
    tokenizedTextLines = tokenizeLines(text)
    model = wordToVec(tokenizedTextLines)    
    # Print first 10 tokenized words and vectorized 10 tokenized words
    for line in tokenizedTextLines[:10]:
        print(line)
        for word in line:
            print(word)
            print(model.wv.__getitem__(word))
    print(model)


if __name__ == "__main__":
    main()