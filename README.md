# RNN

## GENERAL OVERVIEW 

This is the readme for the "RNN.py" file, created by George Barker and Andre Zeromski for CSCI315 Artificial Intelligence with Professor Cody Watson. 

This project is a recurrent neural network (RNN) that functions as a language model.

## Update 1
This assignment represents the data preprocessing segment of an RNN. It begins by initializing the dimensions for the RNN: word_dim is the size of the model’s input layer, hidden_dim is the size of the model’s hidden layer, bptt_truncate is the number of times the model recurs, and U,V, and W are the weight matrices. The model’s first called method is importData, which accepts a file as input and returns the text within it. Next the model calls tokenizeWords, which accepts a string of text and proceeds to substitute punctuation with white space, tokenize each word, change all words to lowercase, filter out stop words, and remove the remaining tokens which are not alphabetic. Once all of the words have been tokenized, the model finds the 8000 most frequently occurring using the method findVocab and returns it as a list. The model will use these 8000 words as its new vocabulary; the next task is to replace those words which occur less frequently with an “UNK” token. In the method tokenizeSentences, the model takes the original string of text and tokenizes it by sentences and removes words according to the 8001 word vocabulary. tokenizeSentences also \adds the tokens “START” and “END” to the beginning and end of each tokenized sentence. Finally, the model calls wordToVec, which creates a Word2Vec model that is saved after training on these tokenized sentences.

### Sources
https://machinelearningmastery.com/clean-text-machine-learning-python/ <br /> 
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/ <br /> 