# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)


# Import, load and unpack  the Fashion MNIST data directly from TensorFlow:
cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images,test_labels) = cifar10.load_data()



#The images are 32x32 NumPy arrays, with pixel values ranging from 0 to 255. The labels are an array of integers,
# ranging from 0 to 9. These correspond to the class of clothing the image represents:



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse','ship', 'truck']


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
plt.imshow(train_images[0],cmap=plt.cm.binary)
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
    plt.ylabel([train_labels[i]])
plt.show()

model = tf.keras.Sequential([                       # api creates model layer by layer
    tf.keras.layers.Flatten(input_shape=(28,28)),   # takes 2d array transforms to 1d
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

model.fit(train_images, train_labels, epochs=10)   # increase epochs to get a higher accuracy , dimishes effectiveness after 30

test_loss, test_acc= model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()]) #The model's linear outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.


predictions = probability_model.predict(test_images)        # grab the predicted set

print(predictions[1])                                        # take a look at the first pridiction probability

print (test_labels[1])                                        # compare with test label to see if correct)


def plot_image(i, predictions_array, true_label, img):       # graphing full set of class predictions
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')


# verify predicitions With the model trained, you can use it to make predictions about some images.
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


#Let's plot several images with their predictions. Note that the model can be wrong even when very confident.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#Finally, use the trained model to make a prediction about a single image.

img = test_images[1]

print(img.shape)

# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. Accordingly, even though you're using a single image, you need to add it to a logits


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# Now predict the correct label for this image:


predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data. Grab the predictions for our (only) image in the batch:


np.argmax(predictions_single[0])