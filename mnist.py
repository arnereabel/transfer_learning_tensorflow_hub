# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.bench import yl

print(tf.__version__)


# Import, load and unpack  the Fashion MNIST data directly from TensorFlow:
fashion_mnist= tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,test_labels) = fashion_mnist.load_data()

#The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. The labels are an array of integers,
# ranging from 0 to 9. These correspond to the class of clothing the image represents:

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# explore the dataset
print(f'train dataset {train_images.shape}')
print(f'test dataset {test_images.shape}')


# as many labels  as images in the dataset
print(f'train_labels as many labels  as images in the dataset {len(train_labels)} each label is an integer in range 0 to 9 {train_labels}')
print(f'test_labels as many labels  as images in the dataset {len(test_labels)} each label is an integer in range 0 to 9 {test_labels}')


# The data must be preprocessed before training the network.
# If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

print(plt.figure())
plt.xlabel('pxl')
plt.ylabel('pxl')
plt.imshow(train_images[0])
plt.colorbar(label=('pxl grayscale values'))
plt.grid(False)
plt.show()


# Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255.
# It's important that the training set and the testing set be preprocessed in the same way:

train_images = train_images / 255.0
test_images = test_images / 255.0

# To verify that the data is in the correct format and that you're ready to build and train the network,
# let's display the first 25 images from the training set and display the class name below each image.

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1) #subplot(nrows, ncols, plot_number)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()