# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:12:30 2019

@author: George Barker and Andre Zeromski
@author: watso
"""
import pickle
import csv
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from collections import Counter 


class RNN:
    
    def __init__(self, wordToVecModel, hidden_dim, word_dim = 10, output_dim = 8003, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.wordToVecModel = wordToVecModel
        self.vocabulary = list(wordToVecModel.model.wv.vocab.keys())
        self.vectorVocabulary = [self.wordToVecModel.model.wv.__getitem__(word) for word in self.vocabulary]
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    
    def feedForward(self, line):
        # Fill the input vector
        hp = np.zeros(self.hidden_dim)
        to = []
        for word in line:
            inputNodes = np.asarray(word)
            hi = np.dot(self.U, inputNodes.transpose())
            hp = np.dot(self.W, hp.transpose())
            hiddenSummation = hi + hp
            ht = np.tanh(hiddenSummation)
            yt = softmax(np.dot(self.V,ht))
            vectorWordIndex = np.argmax(yt)
            to.append(self.vectorVocabulary[vectorWordIndex])
            print(self.vocabulary[vectorWordIndex])
        return to

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)
        

class VectorizeData:
    def __init__(self, data):
        """ Given an array of tokenized words returns a Word2Vec model """
        self.model = Word2Vec(data, size = 10, min_count=1, iter = 5)

    def getNumericFromWord(self, word):
        """ Changes one word token to a vector """
        return self.model.wv.__getitem__(word)
    
    def getNumericFromLine(self, line):
        """ Changes one line of word tokens to a vector of vectors """
        numericSentence = []
        for word in line:
            numericSentence.append(self.getNumericFromWord(word))
        return numericSentence
    
class TokenizeData:
    def __init__(self, filename = "rnnDataset.csv"):
        self.filename = filename
        self.unknown_token = "UNK"
        self.sentence_start_token = "START"
        self.sentence_end_token = "END"
        self.vocabulary_size = 8000
        self.sentences = self.tokenizeLines(self.importData(self.filename))

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
#    data = TokenizeData().getSentences()
#    pickle_out = open("data","wb")
#    pickle.dump(data, pickle_out)
#    pickle_out.close()
#        
#    vdModel = VectorizeData(data)
#    pickle_out = open("vdModel","wb")
#    pickle.dump(vdModel, pickle_out)
#    pickle_out.close()
    
#    # Load the Data
#    pickle_in = open("data","rb")
#    data = pickle.load(pickle_in)
#        
#    # Load the vdModel
    pickle_in = open("vdModel","rb")
    vdModel = pickle.load(pickle_in)
    
#    x_train = []
#    y_train = []
#    
#    dataVector = []
#    for line in data:
#        vectorLine = vdModel.getNumericFromLine(line)
#        dataVector.append(vectorLine)        
#        vectorLine.pop(0)
#        vectorLine.pop()
#        vectorLineCopy = vectorLine.copy()
#        vectorLineCopy.pop()
#        x_train.append(vectorLineCopy)
#        vectorLine.pop(0)
#        y_train.append(vectorLine)
#    
#    pickle_out = open("x_train","wb")
#    pickle.dump(x_train, pickle_out)
#    pickle_out.close()
#    pickle_out = open("y_train","wb")
#    pickle.dump(y_train, pickle_out)
#    pickle_out.close()

#    # Load x_train
#    pickle_in = open("x_train","rb")
#    x_train = pickle.load(pickle_in)
#
#    # Load y_train
#    pickle_in = open("y_train","rb")
#    y_train = pickle.load(pickle_in)
#    
#    pickle_out = open("x_trainShort","wb")
#    pickle.dump(x_train[:10], pickle_out)
#    pickle_out.close()
#    pickle_out = open("y_trainShort","wb")
#    pickle.dump(y_train[:10], pickle_out)
#    pickle_out.close()
    
    # Load x_train
    pickle_in = open("x_trainShort","rb")
    x_train = pickle.load(pickle_in)

    # Load y_train
#    pickle_in = open("y_trainShort","rb")
#    y_train = pickle.load(pickle_in)
    
    ourRNN = RNN(vdModel,100)
    result = ourRNN.feedForward(x_train[1])
    print(result)
    

if __name__ == "__main__":
    main()
