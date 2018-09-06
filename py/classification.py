# Code from: https://www.tensorflow.org/tutorials/keras/basic_classification
# Licence: MIT License

# TensorFlow library and Keras high-level API
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# What version are we on?
print(tf.__version__)

# load the fashion_mnist dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#let's give the labels proper names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#let's look at the data we have
print (train_images.shape)
print (train_labels.shape)
print(len(train_labels))

#pre-process the data (needs to be done before training)
plt.figure()
plt.imshow(test_images[1])
plt.colorbar()
plt.gca().grid(False)
plt.show()

# need to scale the values to go from 0 to 1 for TensorFlow
train_images = train_images / 255.0
test_images = test_images / 255.0

#display the first 25 images
import matplotlib.pyplot as plt
#%matplotlib inline

# plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


plt.show()

# building the neural network requires configuring the layers of the model, then compiling the model
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)), # transforms the format of the images from 2darray to 1d array of 784 pixels
  keras.layers.Dense(128, activation=tf.nn.relu), # Densely-connected, or fully-connected, neural layers. This one has 128 nodes/neurons
  keras.layers.Dense(10, activation=tf.nn.softmax) # a 10-node softmax layer, returns an array of 10 probability scores that sum to 1
]) # each node contains a score that indicates the probability that the current images belongs to one of the 10 classes

# compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), # this is how the model is updated based on the data it sees and its loss function
  loss='sparse_categorical_crossentropy', # this measures how accurate the model is during training, we want to minimize this function to steer the model in the right direction
  metrics=['accuracy']) # used to monitor the training and testing steps. 'Accuracy', the fraction of images that are correctly classified

# train the model
#   feed the training data
#   the model learns to associate images and labels
#   we ask the model to make predictions about the test set and verify them
model.fit(train_images, train_labels, epochs=5)

# as the model trains, the loss and accuracy metrics are displayed. We reach an accuracy of about 0.88 or 88%


# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc) # this is a little less than the accuracy on the training dataset. The gap is an example of 'overfitting', when the ML model performs worse on the new data than training data


#let's make a prediction about a single image
img = test_images[0]

print(img.shape)

#tf.keras models are optimized to make predictions on a colleciton of examples at once, so create a list
img = (np.expand_dims(img,0))

print(img.shape)

predictions = model.predict(img)

print(predictions)
print(class_names[np.argmax(predictions[0])])

predictions = model.predict(test_images)
# let's plot several images with their predictions
# correct are in green, incorrect in red
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)


plt.show()


##########################################################################################################

print(model.inputs)
print(model.outputs)

# save underlying tensorflow model in pb format
tf.saved_model.simple_save(keras.backend.get_session(), "exportdir", inputs={'input': model.inputs[0]}, outputs={'output': model.outputs[0]})

# doesn't save the weights!
#tf.train.write_graph(keras.backend.get_session().graph, ".", "model.pb", as_text=False)

# this is Keras specific and not fully supported in the other languages' bindings to the lower level API
#keras.models.save_model(model, "model.h5", False, True) # saves as HDF5 format
