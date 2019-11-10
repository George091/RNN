# Recurrent Neural Network (RNN)

## GENERAL OVERVIEW 

This is the readme for the "RNN.py" file, created by George Barker and Andre Zeromski for CSCI315 Artificial Intelligence with Professor Cody Watson. 

This project is a recurrent neural network (RNN) that functions as a language model.

## How to run code for Feedforward

Run the python file RNN2.py and the script will load in the tokenized and vectorized data from part 1 using pickle. The data consists of a list with each element being an array of a line which has been tokenized and each word vectorized. The output to running the feedforward on this list will be an array with the word predictions for each word in a line. For simplicity sake we only show the predictions for the first 5 lines. 

If you'd like to load in all data from the scratch you can comment out lines XX and uncomment lines XX. Then run the python file.

## Feedforward (part 2)

Feedforward takes in a list of vectorized tokens and outputs a list where each element is the model's prediction for the most likely output for the given timestep. 

## Data Manipulation (part 1)

This assignment represents the data preprocessing segment of an RNN. The goal of this segment is to embed words from Reddit comments as floating point vectors. These vectors will later allow the model to apply linear and non-linear transformations on them to predict the next token (word) given the previous word embeddings as context. The model creates these embeddings by tokenizing each word in the input text, because tokens can be more easily and methodologically represented as numerical vectors (as opposed to leaving the input text as strings or one hot encodings). These tokenized words then make up rows of text that are also tokenized and passed to a word2vec method, which embeds these tokenized sentences and words with numerical values. The Word2vec encoding creates a vector space such that words appearing in similar contexts will appear nearby in vector space. It will be useful for our linguistic model to represent words in vector space as we want to capture the meaning of a word in our model, not necessarily a specific word.

We were provided code that initialize the dimensions for the RNN: word_dim is the size of the model’s input layer, hidden_dim is the size of the model’s hidden layer, bptt_truncate is the number of times the model recurs for BPTT, and U,V, and W are the weight matrices. These variables will not be called in this segment. The model’s first called method is importData, which accepts a .csv file as input and returns the text within it for manipulation. The text is stored in an array with each element of the array as a line of the .csv.

Next, the model calls tokenizeLines, which accepts rows of text (arrayOfLines) as input. tokenizeLines begins by separating the input text row by row and calls the method tokenizeText on each row. The method tokenizeText takes these rows of text and tokenizes each word and symbol according to NLTK's default word_tokenize method. We want to tokenize these words because tokenizing divides large chunks of texts into more manageable, smaller pieces that can be manipulated into numerical vectors through embedding, which will be useful for finding patterns within the text. Once all words and symbols have been tokenized, we change them all to lowercase and remove stop words (using NLTK's list of stop words), because stop words are typically non-informative of the meaning of the sentence or string of text. It is important to note that changing words to lowercase means we can consider words the same even when they are capitilized at the beginning of a sentence and we can add our own special tokens such as "END". This does however remove meaning from some words. For example, Apple and apple are not the same (company vs food). The goal with these steps are to normalize our input data.

Once each row in tokenizeLine's arrayOfLines has been normalized in this way and returned as a 2D vector (1 column by the number of rows of text from the input file), tokenizeLines proceeds to flatten this vector and calls findVocab. findVocab is a method that takes this tokenized flattened vector and returns the 8000 most frequently occurring words. These 8000 words are then used as our new vocabulary; we want to limit our vocabulary for two reasons. The first is that, according to Zipf's law, we can limit our vocabulary in this way and still retain the same or similar meaning in each sentence. The second reason is if our model's prediction of words is based on the probability of each word occurring next, then a large vocabulary will cause these probabilities to be very small, and thus it is difficult to determine a useful understanding of an output probability. Once  the newly developed vocabulary is created, the model proceeds to take the unflattened arrayOfLines and replaces the words not in the vocabulary with an 'UNK' token.

Each row in arrayOfLines is then manipulated by adding a 'START' and 'END' token to the beginning and end of each line. We do this because it signals to the encoder--decoder structure of the embedding model when to begin encoding and when to terminate to create the final embedding of the given input sequence. Finally, the model calls wordToVec, which creates a Word2Vec model that is saved after training on these tokenized rows of sentences from arrayOfLines. The Word2Vec model contains a total of 8003 words in its vocabulary (8000 most frequent words, "UNK", "START", and "END") and is what is used to embed our tokenized sentences with numerical meaning.

### Sources

"On my honor, I have neither given nor received any unacknowledged aid on this assignment."

https://machinelearningmastery.com/clean-text-machine-learning-python/ <br /> 
https://machinelearningmastery.com/develop-word-embeddings-python-gensim/ <br /> 
https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XcJOey_MyYV <br />
