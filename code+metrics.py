    
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

def database_import():
    """Import and preprocess MNIST dataset"""
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape and flatten the images
    train_images = train_images.reshape((60000, 28*28))
    test_images = test_images.reshape((10000, 28*28))

    # Normalize pixel values to [0,1]
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    def one_hot_encode(labels, num_classes):
        """Convert integer labels to one-hot encoded format"""
        num_labels = len(labels)
        one_hot_matrix = np.zeros((num_labels, num_classes))
        one_hot_matrix[np.arange(num_labels), labels] = 1
        return one_hot_matrix

    # One-hot encode the labels
    train_labels = one_hot_encode(train_labels, 10)
    test_labels = one_hot_encode(test_labels, 10)

    # Return data in correct format (samples, features) - NO TRANSPOSE
    return train_images, train_labels, test_images, test_labels

def initialize_parameters(layer_dims):
    """Initialize weights and biases for the neural network"""
    np.random.seed(42)  # For reproducibility
    parameters = {}
    num_layers = len(layer_dims)
    
    for i in range(1, num_layers):
        # Xavier initialization for better gradient flow
        parameters[f'W{i}'] = np.random.randn(layer_dims[i-1], layer_dims[i]) * np.sqrt(2.0 / layer_dims[i-1])
        parameters[f'b{i}'] = np.zeros((1, layer_dims[i]))
    
    return parameters

def sigmoid(z):
    """Sigmoid activation function with numerical stability"""
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    a = 1 / (1 + np.exp(-z))
    return a, z

def softmax(z):
    """Softmax activation function for output layer"""
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_propagation(X, parameters):
    """Forward propagation through the network"""
    activations = {'A0': X}  # Input layer
    caches = []
    
    num_layers = len(parameters) // 2
    
    # Hidden layers with sigmoid activation
    for i in range(1, num_layers):
        A_prev = activations[f'A{i-1}']
        W = parameters[f'W{i}']
        b = parameters[f'b{i}']
        
        Z = np.dot(A_prev, W) + b
        A, cache_z = sigmoid(Z)
        
        activations[f'A{i}'] = A
        caches.append({
            'A_prev': A_prev,
            'W': W,
            'b': b,
            'Z': cache_z
        })
    
    # Output layer with softmax activation
    A_prev = activations[f'A{num_layers-1}']
    W = parameters[f'W{num_layers}']
    b = parameters[f'b{num_layers}']
    
    Z = np.dot(A_prev, W) + b
    A = softmax(Z)
    
    activations[f'A{num_layers}'] = A
    caches.append({
        'A_prev': A_prev,
        'W': W,
        'b': b,
        'Z': Z
    })
    
    return A, caches

def compute_cost(Y_pred, Y_true):
    """Compute cross-entropy cost"""
    m = Y_true.shape[0]
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    
    cost = -np.sum(Y_true * np.log(Y_pred)) / m
    return cost

def backward_propagation(Y_pred, Y_true, caches):
    """Backward propagation to compute gradients"""
    m = Y_true.shape[0]
    num_layers = len(caches)
    gradients = {}
    
    # Output layer gradients
    dZ = Y_pred - Y_true
    dW = np.dot(caches[-1]['A_prev'].T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    dA_prev = np.dot(dZ, caches[-1]['W'].T)
    
    gradients[f'dW{num_layers}'] = dW
    gradients[f'db{num_layers}'] = db
    
    # Hidden layer gradients
    for i in reversed(range(num_layers - 1)):
        cache = caches[i]
        Z = cache['Z']
        
        # Sigmoid derivative
        sigmoid_z = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        dZ = dA_prev * sigmoid_z * (1 - sigmoid_z)
        
        dW = np.dot(cache['A_prev'].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        if i > 0:  # Not the first layer
            dA_prev = np.dot(dZ, cache['W'].T)
        
        gradients[f'dW{i+1}'] = dW
        gradients[f'db{i+1}'] = db
    
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    """Update parameters using gradients"""
    num_layers = len(parameters) // 2
    
    for i in range(1, num_layers + 1):
        parameters[f'W{i}'] -= learning_rate * gradients[f'dW{i}']
        parameters[f'b{i}'] -= learning_rate * gradients[f'db{i}']
    
    return parameters

def train_model(X, Y, layer_dims, epochs, learning_rate):
    """Train the neural network"""
    parameters = initialize_parameters(layer_dims)
    cost_history = []
    
    for epoch in range(epochs):
        # Forward propagation
        Y_pred, caches = forward_propagation(X, parameters)
        
        # Compute cost
        cost = compute_cost(Y_pred, Y)
        cost_history.append(cost)
        
        # Backward propagation
        gradients = backward_propagation(Y_pred, Y, caches)
        
        # Update parameters
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Cost: {cost:.4f}")
    
    return parameters, cost_history

def evaluate_model(parameters, X_test, Y_test):
    """Evaluate the trained model"""
    print("\n--- Model Evaluation ---")
    
    # Make predictions
    Y_pred_probs, _ = forward_propagation(X_test, parameters)
    
    # Convert to class predictions
    Y_pred_classes = np.argmax(Y_pred_probs, axis=1)
    Y_true_classes = np.argmax(Y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(Y_true_classes, Y_pred_classes)
    precision = precision_score(Y_true_classes, Y_pred_classes, average='macro', zero_division=0)
    recall = recall_score(Y_true_classes, Y_pred_classes, average='macro', zero_division=0)
    f1 = f1_score(Y_true_classes, Y_pred_classes, average='macro', zero_division=0)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Average Precision: {precision:.4f}")
    print(f"Macro Average Recall: {recall:.4f}")
    print(f"Macro Average F1-Score: {f1:.4f}")
    
    # Detailed classification report
    print("\n--- Classification Report ---")
    print(classification_report(Y_true_classes, Y_pred_classes, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(Y_true_classes, Y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def predict_single_image(parameters, X_test, Y_test):
    """Predict and display a random test image"""
    # Select random image
    random_idx = np.random.randint(0, X_test.shape[0])
    random_image = X_test[random_idx:random_idx+1]  # Keep 2D shape
    true_label = Y_test[random_idx:random_idx+1]
    
    # Make prediction
    prediction_probs, _ = forward_propagation(random_image, parameters)
    predicted_digit = np.argmax(prediction_probs)
    true_digit = np.argmax(true_label)
    
    # Display image
    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(random_image.reshape(28, 28), cmap='gray')
    plt.title(f'True: {true_digit}')
    plt.axis('off')
    
    # Display prediction probabilities
    plt.subplot(1, 2, 2)
    plt.bar(range(10), prediction_probs[0])
    plt.title(f'Predicted: {predicted_digit}')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.show()
    
    print(f"True digit: {true_digit}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Prediction confidence: {prediction_probs[0][predicted_digit]:.4f}")

# Main execution
if __name__ == "__main__":
    # Load data
    train_images, train_labels, test_images, test_labels = database_import()
    
    # Define network architecture
    layer_dims = [784, 128, 10]  # Input: 784, Hidden: 128, Output: 10
    
    # Set hyperparameters
    epochs = 1000
    learning_rate = 0.1
    
    print("Training neural network...")
    print(f"Training data shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Network architecture: {layer_dims}")
    
    # Train the model
    trained_parameters, cost_history = train_model(
        train_images, train_labels, layer_dims, epochs, learning_rate
    )
    
    # Plot training cost
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost Over Time')
    plt.grid(True)
    plt.show()
    
    # Evaluate the model
    evaluate_model(trained_parameters, test_images, test_labels)
    
    # Test on a single image
    print("\n--- Single Image Prediction ---")
    predict_single_image(trained_parameters, test_images, test_labels)

