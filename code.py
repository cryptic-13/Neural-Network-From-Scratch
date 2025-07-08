import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Importing the database required to train the machine learning model:
def database_import():
  from keras.datasets import mnist #this will load the database into the model
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  # Reshape and flatten the images uploaded:
  train_images = train_images.reshape((60000, 28*28))
  test_images = test_images.reshape((10000, 28*28))

  # Pixel values should be between 0 and 1 to feed into the neuron processes
  train_images = train_images.astype('float32')/255
  test_images = test_images.astype('float32')/255

  """we need to rephrase the values into a way that is easier to understand for the model, more binary based values.
  a method to do this is called 'one-hot encoding', which essentially is a process that takes data that is categorized and
  converts it into a list of 10 elements, each element representing a number from 0 to 9. 7 can be represented as [0,0,0,0,0,0,0,1,0,0].
  To do this we use a utility present in keras called 'to_categorical' or we can write it from scratch"""

  def one_hot_encode(labels, num_classes):
    """
    Converts a list of integer labels into a one-hot encoded NumPy array.

    Parameters:
        labels: A list or NumPy array of integer labels.
        num_classes: The total number of unique classes.

    Returns:
        A NumPy array where each row is the one-hot encoded representation
        of the corresponding label.
    """
    num_labels = len(labels)
    # Create an array of zeros with shape (num_labels, num_classes)
    one_hot_matrix = np.zeros((num_labels, num_classes))

    # Use advanced indexing to set the appropriate elements to 1
    one_hot_matrix[np.arange(num_labels), labels] = 1

    return one_hot_matrix

  # One-hot encode the labels using the custom function
  train_labels = one_hot_encode(train_labels, 10)
  test_labels = one_hot_encode(test_labels, 10)


  return train_images.T, train_labels.T, test_images.T, test_labels.T


#as there is no input from back propagation for the first trial, the parameters must be randomly initialized
# parameters --> weight matrix and bias vector

def initial_parameters(layer_dims) :
    np.random.seed(3)
    parameters= {}
    numLayers= len(layer_dims)
     #numLayersL= number of layers in the network, layer_dims is a list of dimensions of each layer
    for x in range(1, numLayers):
        parameters['W'+str(x)]= np.random.randn(layer_dims[x], layer_dims[x-1])*0.01
        parameters['b'+str(x)]= np.zeros((layer_dims[x],1))
    return parameters


    #define sigmoid function, which takes the input from previous layer
    #  --> ΣW(i)A(i) +B (for i ranges from 1 to total number of neurons in the previous layer)
    # W(i) is the corresponding element (matched by 'i' value) in the weight matrix and
    # A(i) is the activation level of the ith neuron in the previous layer
    # to find the activation level of a neuron in the current layer, sigmoid takes this summation and squishes it to a value bw 0 and 1


def sigmoid(z):
        A= 1/(1 +(np.exp(-z)))
        cache= z
        return (A,cache)

    #cache is returned to facilitate future backpropagation
    #In forward propagation, the first layer takes its input from the database, processes its output using parameters and input
    #output is passed to next layer and so on


def forward_prop(input, parameters) :
        A= input
        #assign the input to activation level of first layer of neurons
        caches= []
        neuronCount= len(parameters)//2
        #because for each neuron, 2 parameters exist, weight and bias
        for x in range (1, neuronCount+1) :
            A_prev= A
            #store the initial input to first layer neuron
            z= np.dot( parameters['W'+str(x)] ,A_prev ) + parameters['b'+str(x)]
            # z represents the activation of each neuron in the first layer, after squishification using sigmoid fn
            # this is the output of the first layer, using the linear hypothesis formula

            linear_cache= (A_prev, parameters['W' + str(x)], parameters['b' + str(x)])


            A, active_cache= sigmoid(z)

            cache= (linear_cache, active_cache)
            caches.append(cache)

            #linear_cache contains all the individual components that calculated the expression to which sigmoid was appplied
            #linear_cache stores the activation levels, weights and biases as 3 separate entities
            #activation cache store the linear hypothesis post matrix multiplication -->  ΣW(i)A(i) +B --> this value
        return A, caches


def cost_fn(predicted, truth) :
                A= predicted
                #a matrix of all the final output (predictions) made by the neural network
                #truth is a vector/matrix of all the actual values or true values taken from the training dataset
                #size= number of data points in training dataset

                size= truth.shape[1]
                # Use element-wise multiplication instead of dot product
                cost= (-1/size) * np.sum(np.multiply(truth, np.log(predicted)) + np.multiply(1-truth, np.log(1-predicted)))
                return cost


def chain_rule_logic(dA,cache):
                linear_cache, active_cache= cache
                z= active_cache
                s = 1 / (1 + np.exp(-z)) # Recalculate sigmoid for the derivative
                dZ=  dA*s * (1-s)
                #derivative of sigmoid function wrt lienar output of that neuron

                A_prev, W,b= linear_cache
                count= A_prev.shape[1]
                #count = number of neurons in previous layer
                dW= (1/count) * (np.dot(dZ, A_prev.T))
                db= (1/count) * np.sum(dZ, axis=1, keepdims=True)
                dA_prev= np.dot (W.T, dZ)

                #dw, db, dA are the derivatives of the cost function wrt weights, biases and previous activations.

                return (dA_prev, dW, db)


def backprop (A_last, truth, caches) :
    #A_last is a vector of activations of the last layer of neurons, ergo the networks final predictions
    # truth is the vector of actual outputs, or true labels

    gradients = {}
    layerCount= len (caches)
    final_neuron_count= A_last.shape[1]
    truth = truth.reshape(A_last.shape)

    dAL= -(np.divide(truth, A_last) - np.divide(1-truth, 1-A_last))
    current_cache=  caches[layerCount-1]
    L= layerCount

    #create a dictionary of all the gradients calculated for backpropagation
    # each gradient is a partial derivative of the cost fn wrt each parameter
    # each layer will have a different set of A1, W1, b1 --> parameters and activation
    # this you can get from its 'caches'
    # so, for each layer, 3 gradients are computed and stored as a tuple.
    # this process is repeated for as many layers as there are

    gradients['dA'+str(L-1)], gradients['dW'+ str(L)], gradients['db'+str(L)] = chain_rule_logic(dAL, current_cache)

    for l in reversed(range(L-1)):
        current_cache= caches[l]
        dA_prev_temp, dW_temp, db_temp= chain_rule_logic(gradients['dA'+str(l+1)], current_cache)
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

    return (gradients)


def update_parameters (parameters, gradients, learning_rate) :

    L = len(parameters) // 2

    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] -learning_rate*gradients['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] -  learning_rate*gradients['db'+str(l+1)]

    return parameters


def train(X, Y, layer_dims, epochs, lr):
    parameters = initial_parameters(layer_dims)
    cost_history = []

    for i in range(epochs):
        Y_hat, caches = forward_prop(X, parameters)
        cost = cost_fn(Y_hat, Y)
        cost_history.append(cost)
        gradients = backprop(Y_hat, Y, caches)

        parameters = update_parameters(parameters, gradients, lr)

        if i % 100 == 0:
            print(f"Cost after epoch {i}: {cost}")


    return parameters, cost_history

# Assuming you have imported Keras and TensorFlow if running in Colab
# !pip install tensorflow # Already installed in Colab
# !pip install keras # Already installed in Colab
# import tensorflow as tf # Already imported
# from tensorflow import keras # Already imported


# Load the MNIST data
train_images, train_labels, test_images, test_labels = database_import()

# We define the number of neurons in each layer
# layer_dims will be defined with the same number of arguments as the number of layers we want, using the shape of train_images
# and shape of train_labels as our beginning and ending layers
layer_dims = [train_images.shape[0], 128, train_labels.shape[0]]

# Set hyperparameters
epochs = 1000
l_r = 0.01

# Train the model
trained_parameters, costs = train(train_images, train_labels, layer_dims, epochs, l_r)

# Plot the cost history
plt.plot(np.squeeze(costs)) # Use np.squeeze to remove single-dimensional entries
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost Reduction over Epochs")
plt.show()

# Select a random image from the test set
random_index = np.random.randint(0, test_images.shape[1])
random_image = test_images[:, random_index].reshape(-1, 1) # Select the image and reshape to (features, 1)
true_label = test_labels[:, random_index].reshape(-1, 1) # Select the corresponding true label

# Display the image
# Reshape the flattened image (784 features) back to 28x28 for displaying
image_to_display = random_image.reshape(28, 28)
plt.imshow(image_to_display, cmap='gray')
plt.title(f"True Label (One-Hot Encoded):\n{true_label.T}")
plt.axis('off')
plt.show()

# Feed the image into the trained model for prediction
# The forward_prop function expects input with shape (features, num_samples)
# Our random_image is already (features, 1)
prediction, _ = forward_prop(random_image, trained_parameters)

# Interpret the prediction
# The output 'prediction' will be a vector of probabilities (one-hot encoded like)
# The predicted digit is the one with the highest probability
predicted_digit = np.argmax(prediction)

print(f"Model Prediction: {predicted_digit}")

# You can compare the predicted_digit with the true label (after decoding the one-hot encoding)
true_digit = np.argmax(true_label)
print(f"True Digit: {true_digit}")
