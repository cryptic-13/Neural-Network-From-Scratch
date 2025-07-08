
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow import keras # This line is no longer needed as we'll use tf.keras directly
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns # For a nicer confusion matrix plot

# Importing the database required to train the machine learning model:
def database_import():
    # Use tf.keras.datasets to ensure compatibility with the TensorFlow installation
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

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


# Define sigmoid function for hidden layers
def sigmoid(z):
    A= 1/(1 +(np.exp(-z)))
    cache= z
    return (A,cache)

# Define softmax function for the output layer (multi-class classification)
def softmax(z):
    # Subtract max for numerical stability to prevent overflow with large exponents
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    A = exp_z / np.sum(exp_z, axis=0, keepdims=True)
    cache = z
    return A, cache


def forward_prop(input, parameters) :
    A= input
    #assign the input to activation level of first layer of neurons
    caches= []
    neuronCount= len(parameters)//2 # Total number of layers with weights/biases

    for x in range (1, neuronCount+1) :
        A_prev= A
        # Linear hypothesis: Z = W * A_prev + b
        z= np.dot( parameters['W'+str(x)] ,A_prev ) + parameters['b'+str(x)]

        linear_cache= (A_prev, parameters['W' + str(x)], parameters['b' + str(x)])

        # Apply activation function: Softmax for the last layer, Sigmoid for hidden layers
        if x == neuronCount: # This is the last layer
            A, active_cache = softmax(z)
        else: # This is a hidden layer
            A, active_cache = sigmoid(z)

        cache= (linear_cache, active_cache)
        caches.append(cache)

    return A, caches


def cost_fn(predicted, truth) :
    size= truth.shape[1]
    # Categorical Cross-Entropy Loss
    # Clip predicted values to avoid log(0) or log(1) issues, which can lead to NaN/inf
    predicted = np.clip(predicted, 1e-10, 1 - 1e-10)
    cost= (-1/size) * np.sum(truth * np.log(predicted))
    return cost


def chain_rule_logic(dA,cache):
    linear_cache, active_cache= cache
    z= active_cache
    s = 1 / (1 + np.exp(-z)) # Recalculate sigmoid for its derivative
    dZ=  dA*s * (1-s) # Derivative of sigmoid function wrt linear output of that neuron

    A_prev, W,b= linear_cache
    count= A_prev.shape[1] # Number of samples in the batch
    
    dW= (1/count) * (np.dot(dZ, A_prev.T))
    db= (1/count) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev= np.dot (W.T, dZ)

    # dw, db, dA_prev are the derivatives of the cost function wrt weights, biases and previous activations.
    return (dA_prev, dW, db)


def backprop (A_last, truth, caches) :
    # A_last is a vector of activations of the last layer of neurons (network's final predictions)
    # truth is the vector of actual outputs, or true labels (one-hot encoded)

    gradients = {}
    layerCount= len (caches) # Total number of layers with weights/biases

    # Derivative of Categorical Cross-Entropy with Softmax for the last layer
    # This simplifies directly to (A_last - truth)
    dAL = A_last - truth

    # Backpropagate for the last layer (L)
    current_cache_L =  caches[layerCount-1] # Cache for the last layer
    A_prev_L, W_L, b_L = current_cache_L[0] # linear_cache for the last layer
    count_L = A_prev_L.shape[1] # Number of samples

    gradients['dW'+ str(layerCount)] = (1/count_L) * (np.dot(dAL, A_prev_L.T))
    gradients['db'+str(layerCount)] = (1/count_L) * np.sum(dAL, axis=1, keepdims=True)
    gradients['dA'+str(layerCount-1)] = np.dot (W_L.T, dAL)

    # Backpropagate through hidden layers (from L-1 down to 1)
    for l in reversed(range(layerCount-1)): # Iterate from second to last layer down to first hidden layer (index 0)
        current_cache = caches[l] # Cache for the current layer (l+1 in terms of W/b index)
        # dA_prev_temp here is the dA for the current layer's activation (which is the input to the next layer's backprop)
        dA_prev_temp, dW_temp, db_temp = chain_rule_logic(gradients['dA'+str(l+1)], current_cache)
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

    return (gradients)


def update_parameters (parameters, gradients, learning_rate) :

    L = len(parameters) // 2 # Number of layers with learnable parameters

    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * gradients['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * gradients['db'+str(l+1)]

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

def evaluate_model(parameters, X_test, Y_test):
    """
    Evaluates the trained neural network model on the test set.

    Parameters:
        parameters (dict): Dictionary of trained weights and biases.
        X_test (np.array): Test images (features).
        Y_test (np.array): True labels for test images (one-hot encoded).

    Returns:
        None: Prints and plots evaluation metrics.
    """
    print("\n--- Model Evaluation ---")

    # Make predictions on the test set
    Y_pred_raw, _ = forward_prop(X_test, parameters)

    # Convert one-hot encoded predictions and true labels to class labels
    Y_pred_classes = np.argmax(Y_pred_raw, axis=0)
    Y_true_classes = np.argmax(Y_test, axis=0)

    # 1. Accuracy
    accuracy = accuracy_score(Y_true_classes, Y_pred_classes)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # 2. Precision, Recall, F1-score (per class and averaged)
    # Using 'macro' average for unweighted mean, 'weighted' for weighted mean by support
    precision = precision_score(Y_true_classes, Y_pred_classes, average='macro', zero_division=0)
    recall = recall_score(Y_true_classes, Y_pred_classes, average='macro', zero_division=0)
    f1 = f1_score(Y_true_classes, Y_pred_classes, average='macro', zero_division=0)

    print(f"Macro Average Precision: {precision:.4f}")
    print(f"Macro Average Recall: {recall:.4f}")
    print(f"Macro Average F1-Score: {f1:.4f}")

    # Detailed report per class
    print("\n--- Classification Report (Per Class) ---")
    from sklearn.metrics import classification_report
    print(classification_report(Y_true_classes, Y_pred_classes, zero_division=0))

    # 3. Confusion Matrix
    cm = confusion_matrix(Y_true_classes, Y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# Load the MNIST data
train_images, train_labels, test_images, test_labels = database_import()

# We define the number of neurons in each layer
# layer_dims will be defined with the same number of arguments as the number of layers we want, using the shape of train_images
# and shape of train_labels as our beginning and ending layers
layer_dims = [train_images.shape[0], 128, train_labels.shape[0]] # Input, Hidden, Output

# Set hyperparameters
epochs = 1000
l_r = 0.01

# Train the model
trained_parameters, costs = train(train_images, train_labels, layer_dims, epochs, l_r)

# Plot the cost history
plt.figure(figsize=(10, 6))
plt.plot(np.squeeze(costs)) # Use np.squeeze to remove single-dimensional entries
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost Reduction over Epochs")
plt.grid(True)
plt.show()

# --- Evaluate the model on the test set ---
evaluate_model(trained_parameters, test_images, test_labels)


# Select a random image from the test set
random_index = np.random.randint(0, test_images.shape[1])
random_image = test_images[:, random_index].reshape(-1, 1) # Select the image and reshape to (features, 1)
true_label = test_labels[:, random_index].reshape(-1, 1) # Select the corresponding true label

# Display the image
# Reshape the flattened image (784 features) back to 28x28 for displaying
image_to_display = random_image.reshape(28, 28)
plt.figure(figsize=(4, 4))
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

print(f"Model Prediction for Random Image: {predicted_digit}")

# You can compare the predicted_digit with the true label (after decoding the one-hot encoding)
true_digit = np.argmax(true_label)
print(f"True Digit for Random Image: {true_digit}")

