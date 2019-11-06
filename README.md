# RNN

## GENERAL OVERVIEW 

This is the readme for the "RNN.py" file, created by George Barker and Andre Zeromski for CSCI315 Artificial Intelligence with Professor Cody Watson. 

This project is a recurrent neural network (RNN) that functions as a language model.

## Data Manipulation

This assignment represents the data preprocessing segment of an RNN. We were provided code that initialize the dimensions for the RNN: word_dim is the size of the model’s input layer, hidden_dim is the size of the model’s hidden layer, bptt_truncate is the number of times the model recurs for BPTT, and U,V, and W are the weight matrices. The model’s first called method is importData, which accepts a file as input and returns the text within it. Next the model calls tokenizeWords, which accepts a string of text and proceeds to remove all punctuation with an empty character (except for apostrophes), tokenize each word based on whitespace separating words, change all words to lowercase, filter out stop words, remove remaining apostrophes, and remove the remaining tokens which are not alphabetic. Once all of the words have been tokenized, the model finds the 8000 most frequently occurring using the method findVocab and returns it as a list. The model will use these 8000 words as its new vocabulary; the next task is to replace those words which occur less frequently with an “UNK” token. In the method tokenizeSentences, the model takes the original string of text and tokenizes it by sentences and replaces words with "UNK" not in the 8000 word vocabulary. The tokenizeSentences function also adds the tokens “START” and “END” to the beginning and end of each tokenized sentence. Finally, the model calls wordToVec, which creates a Word2Vec model that is saved after training on these tokenized sentences.

### Sources
https://machinelearningmastery.com/clean-text-machine-learning-python/ <br /> 
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/ <br /> 
https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XcJOey_MyYV <br />