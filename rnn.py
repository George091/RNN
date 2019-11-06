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
import string
from collections import Counter 
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer


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
    model = Word2Vec(sentences, min_count=1, iter = 5)
    words = list(model.wv.vocab)
#    print(words)
    print(model['START'])
#    print(model)
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
    # replace periods with a space
    text=re.sub("[.]", " ", text)
    # remove punctuation except apostrophe
    text = re.sub("[^\w\d'\s]+",'',text)
    # tokenize words based on seperating whitespace
    tokens = WhitespaceTokenizer().tokenize(text)
    # lowercase all words
    tokens = [token.lower() for token in tokens]
    # remove stop words
    stopWords = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stopWords]
    # remove remaining apostrophes
    table = str.maketrans('', '', string.punctuation)
    stripped = [word.translate(table) for word in tokens]
    # remove non-alphabetic tokens
    words = [word for word in stripped if word.isalpha()]
    return words
    
def findVocab(listOfWords):
    """ Take in a tokenized array of words and returns a list of top 8000 words """
    counterWords = Counter(listOfWords)
    most_occur = counterWords.most_common(8000)
    vocab = []
    for position in most_occur:
        vocab.append(position[0])
#    print(most_occur)
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
    tokenizedSentences = tokenizeSentences(text, vocabulary)
    embedding = wordToVec(tokenizedSentences)
#    print(len(vocabulary))
#    print(tokenizedSentences[:100])
#    print(set(embedding) ^ set(vocabulary))

if __name__ == "__main__":
    main()