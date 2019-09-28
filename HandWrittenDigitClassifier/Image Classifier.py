import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# print(tf.__version__) # prints the version of tensorflow

# prepare the data
mnist = tf.keras.datasets.mnist

# x_train represents the features which in this case are the pixel values of the 28 x 28 images of the digits 0-9
# y_train is the label - is the image a 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# looking at the input data (the grey scale values that make up the image)
# print(x_train[0])

# visualizing the number
# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()

# viewing the label associated with the image for x_train[0]
# print(y_train[0])

# normalize the data. This the RGB values which are [0, 255] inclusive to some value between 0 and 1 or -1 and 1
# by normalizing the data it makes it easier to do calculations and operations
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# viewing the internal data of the image we can see that we have modified the pixels values
# print(x_train[0])

# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()

# building the model

# A sequential model means things are going to move in a direct order and this model is a feed forward model 
model = tf.keras.models.Sequential()

# Since the image is a 28 x 28 grid, we have to manipulate the image such that we can load it into a column vector with 784 neurons
# flattening does this conversion
model.add(tf.keras.layers.Flatten()) # flattening the first layer of the neural network

# define the hidden layers of the network (the hidden layers are the layers n between the first and last layers of the network)
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # in this second layer or first layer since we are using 0 based counting, there are 128 neurons and we have decided to pick the RELU function as the activation functon

# the output layer - contains only 10 neurons where each neuron corresponds to some digit between [0, 9] inclusive
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) # the softmax function is used because we are trying to get a probability distribution of which of the possible prediction options is the the we're passing features through of is

# compile the model - this is where we pass the settings for optimizing and training the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) # adam is an efficient stochastic gradient descent algorithm

model.fit(x_train, y_train, epochs = 3)

# test on out of sample data
val_loss, val_acc = model.evaluate(x_test, y_test)

# print(val_loss)
# print(val_acc)

# save the model
model.save('epic_num_reader.model')

# load the model back
new_model = tf.keras.models.load_model('epic_num_reader.model')

# predict stuff
predictions = new_model.predict(x_test)

# print(predictions)

# convert the prediction (currently a distribution) into something that can be understood
print(np.argmax(predictions[42]))

# look at the actual test input for image[0]

plt.imshow(x_test[42], cmap = plt.cm.binary)
plt.show()



