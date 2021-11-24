import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import datetime



#get pre trained models from hub
mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
mobilenet_v3_large = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5"

#set wich model to use
classifier_model = mobilenet_v2

IMAGE_SHAPE = (224, 224)


classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])

#dwonload the dataset
data_root = tf.keras.utils.get_file(
  'flower_photos',
  'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)



# load the data into the model from the dataset off harddisk and geenerate a tf.data.Dataset
batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

#check how many classes the dataset has
class_names =  np.array(train_ds.class_names)
print(class_names)  #['daisy' 'dandelion' 'roses' 'sunflowers' 'tulips']


# tf hub convention expects float inputs [0,1] range use the rescaling preprocessing KerasLayer
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) #Where x—images, y—labels.
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

#finish the input pipeline by using buffered prefetching with Dataset.prefetch,
# so you can yield the data from disk without I/O blocking issues.
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


#Now, run the classifier on an image batch:
result_batch = classifier.predict(train_ds)

predicted_class_names = imagenet_labels[tf.math.argmax(result_batch, axis=-1)]
print(predicted_class_names)


#Check how these predictions line up with the images:
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")