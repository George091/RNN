# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:12:30 2019

@author: George Barker and Andre Zeromski
@author: watso
"""

import csv
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from collections import Counter 


class RNN:
    
    def __init__(self, word_dim = 10, hidden_dim, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    
##    def feedForward(self, embedding):
##        # Fill the input vector
##        prev_hidden = np.zeros(hidden_dim)
##        for word in embedding:
##            
##        self.inputNodes = model[i]
##        # Hidden nodes 1 (unsquashed and squashed)
##        self.hiddenNodes1 = np.dot(self.U, self.inputNodes)
##        self.aHiddenNodes1 = self.sigmoid(self.hiddenNodes1)
##        # Hidden nodes 2 (unsquashed and squashed)
##        self.hiddenNodes2 = np.dot(self.weights23, self.aHiddenNodes1)
##        self.aHiddenNodes2 = self.sigmoid(self.hiddenNodes2)
##        # Output nodes (unsquashed and squashed)
##        self.outputNodes = np.dot(self.weights34, self.aHiddenNodes2)
##        self.aOutputNodes = self.sigmoid(self.outputNodes)

class embedding:
    
    def __init__(self):
        """ Given an array of tokenized words returns a Word2Vec model """
        self.model = Word2Vec(tokenizeData().getSentences(), size = 10, min_count=1, iter = 5)

    def getNumericFromWord(self, word):
        return self.model.wv.__getitem__(word)

class tokenizeData:
    
    def __init__(self, filename = "rnnDataset.csv"):
        self.filename = filename
        self.sentences = self.tokenizeLines(self.importData(self.filename))
        # Data cleaning parameters
        self.unknown_token = "UNK"
        self.sentence_start_token = "START"
        self.sentence_end_token = "END"
        self.vocabulary_size = 8000

    def getSentences(self):
        return self.sentences
    
    def tokenizeLines(self, arrayOfLines):
        """ Takes in a array of lines of text and returns tokenized lines """
        i = 0
        for line in arrayOfLines:
            arrayOfLines[i] = self.tokenizeText(line[0])
            i += 1
        flattened  = [val for sublist in arrayOfLines for val in sublist]
        vocabulary = self.findVocab(flattened)
        l = 0
        for line in arrayOfLines:
            w = 0
            for word in line:
                if word not in vocabulary:
                    arrayOfLines[l][w] = self.unknown_token
                w += 1
            arrayOfLines[l].insert(0, self.sentence_start_token)
            arrayOfLines[l].append(self.sentence_end_token)
            l += 1
        return arrayOfLines

    def tokenizeText(self, text):
        """ Takes in a string of text and returns an array of tokenized words """
        # tokenize words based on seperating whitespace
        tokens = word_tokenize(text)
        # lowercase all words
        tokens = [token.lower() for token in tokens]
        # remove stop words
        stopWords = set(stopwords.words('english'))
        tokens = [word for word in tokens if not word in stopWords]
        return tokens
        
    def findVocab(self, listOfWords):
        """ Take in a tokenized array of words and returns a list of top 8000 words """
        counterWords = Counter(listOfWords)
        most_occur = counterWords.most_common(self.vocabulary_size)
        vocab = []
        for position in most_occur:
            vocab.append(position[0])
        return vocab
        
    def importData(self, filename):
        """ Return the text from the file where each line is an element in an array """
        f = open(filename)
        csv_f = csv.reader(f)
        text = []
        for row in csv_f:
            text.append(row)
        return text

def main():
    data = tokenizeData().getSentences()
    

if __name__ == "__main__":
    main()
