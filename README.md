- We limit the RNN to 8000 vocabulary
	- we can represent 80 - 90% of the dataset taking the most frequent words
	- 

Why max_review_length of 500 works really well (perhaps sentiment can be obtained,for most reviews, within the first part; provide overall movie impression in the beginning.
What functions does and how - paragraph

Dropout of .3 means 30% of the time we will be knocking out particular nodes. Dropout goes in between output layer and the previous hidden layer.

Dense layer is a fully connected layer. With # nodes. 

Compile creates the model. 

Adam optimizer. Optimizer is way we tune the weights. (previous stochastic gradient descent). Adam is a spin off that also changes the learning rate. Use momentum variable to change learning rate.

fit trains the model.

