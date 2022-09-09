

# Deep-Learning


A Deep learning programming project where we have to develop a Neural Network with Back Propagation using gradient descent Algorithm which should be flexible to tune with different Hyperparameter configurations

## Functions 
#### NeuralNetwork() 
- layers_size : number of nodes present in the each layer in the form of a list (ex: at 0 th index we have number of features in input data)
- epochs : Number of iterations we want to run the algorithm on given dataset (default = 5)
- learning_rate : learning rate of the model (default = 0.01)
- l2_lambda : weight decay parameter (default = 0)
- optimizer : Optimization Algorithm to update weights (default = "standard) (options = "standard" , "sgd", "momentumGD", "nesterov" ,"rmsprop", "adam", "nadam")
- activation : Activation Function for hidden layers (default = "sigmoid") (options = "relu", "sigmoid", "tanh")
- wtype : Weights initialization method (default = "random") (options = "random", "xavier" )
- loss : loss function to calculate loss (default = "cross_entropy") (options = "cross_entropy" , "mean_square")

#### fit()
 - x_train : Feature dataset to train the Neural Network
 - y_train : Labels Dataset to train the Neural Network
 - x_test  : Feature dataset to test the neural network
 - y_test  : Class label dataset to test the accuracy of neural network
 - batch_size : Batch size to train the dataset (default = 64)
 
 - ##### output
   - displays Validation dataset loss, avalidation dataset accuracy, test dataset loss and test data accuracy
   - return prediction on test dataset-
   - logs the data to your wandb panel
   
## Hyperparameter configurations used
 - number of epochs: 5, 10
 - number of hidden layers:  3, 4, 5
 - size of every hidden layer:  32, 64, 128
 - weight decay (L2 regularisation): 0, 0.0005,  0.5
 - learning rate: 1e-3, 1 e-4 
 - optimizer:  sgd, momentum, nesterov, rmsprop, adam, nadam
 - batch size: 16, 32, 64
 - weight initialisation: random, Xavier
 - activation functions: sigmoid, tanh, ReLU 

#### Training the model
 - to train the model we need to initalise the Neural network with NeuralNetwork() function and Train the model using fit() method
 
#### Evaluating the model
 - to evaluate the model just pass the test data to fit() function it will display the validation loss, validation accuracy, test loss and test accuracy
