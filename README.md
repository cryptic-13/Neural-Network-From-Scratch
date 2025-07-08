# Neural-Network-From-Scratch
## Using Numpy, Matplotlib and Tensorflow to build a neural network from scratch on the MNIST Database. 

Reading through this project demonstartates a thorough breakdown of how neural networks, receive, process and train on data from a dataset to predict outputs. 


This code works on the MNIST database, containing images of handwritten digits, which the model trains to read and identify. 


It doesnt recognize shapes, or characteristic parts of the numbers, like say, narrowing the possible numbers down to 2,3,5,6,8,9 or 0 as a curved feature was present and so on. 
It makes predictions based on pixel brightness and activates neurons in each layer accordingly. The activation level of each neuron in a particular layer plays a role in deciding the activation level of every other neuron in the next layer. This is known is forward propagation. 


The final layer consists as many neurons as there are total number of possible outputs, for recognizing a digit, there are 10 possible outcomes (0-9) and thus the final layer has 10 neurons. The one with the maximum acitvation out of these 10 neurons becomes the predicted number. 


While training the neural network, if the predicted number does not match the actual label, back propagation adjusts the weights and biases such that the correct prediction is made. 
After training on all the examples in the dataset, the weights and biases which produce correct identification for maximum number of training examples are selected as the 'optimized parameters' and a prediction made on any new input (not part of the training dataset) is made using these final parameters. 


New inputs are used to gauge the performance of the model (based on accuracy and indications of overfitting) to see how well what the neural network has learned from training applies to new data inputs. 
Use the following flowchart to visualze how a neural network works in steps. 


![image](https://github.com/user-attachments/assets/79867849-45e5-4433-be36-ed54f36233cc)

The program makes the output more comprehendable by printing the image taken from the training dataset, displaying its true label and the model's prediction. This is a more layman and visual way of understanding how the model works rather than printing a bunch of technical perofmance metrics. 
