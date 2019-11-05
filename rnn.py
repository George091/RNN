# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:12:30 2019

@author: watso
"""

import csv
import numpy as np
import nltk
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
        

def main():
    vocabulary_size = 8000
    unknown_token = "UNK"
    sentence_start_token = "START"
    sentence_end_token = "END"