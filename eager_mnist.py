from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow.contrib.eager as tfe
import dropalan

import numpy as np

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

tf.enable_eager_execution()

def get_mnist_dataset(bsize):
  # Fetch and format the mnist data
  (mnist_images, mnist_labels), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

  print("Shape of training images is: " + str(mnist_images.shape))

  approx_mean = np.mean(mnist_images[:1000,:,:])

  print("mean is: " + str(approx_mean))

  mnist_images = (mnist_images.astype(np.float32).reshape(-1, 28*28) - approx_mean ) / 255.0
  X_test = (X_test.astype(np.float32).reshape(-1, 28*28) - approx_mean ) / 255.0

  napprox_mean = np.mean(mnist_images[:1000,:])

  print("new mean is: " + str(napprox_mean))

  dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images, tf.float32),
     tf.cast(mnist_labels,tf.int64)))
  dataset = dataset.shuffle(100).batch(bsize)

  testdataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_test, tf.float32),
     tf.cast(y_test,tf.int64)))
  testdataset = testdataset.shuffle(100).batch(bsize)

  return dataset,testdataset

#converts a batched RGB dataset of images to grayscale using the luminosity formula I randomly found by googling
def convert_to_grayscale(x):
  return 0.21*x[:,:,:,0] + 0.72*x[:,:,:,1] + 0.07*x[:,:,:,2]

def get_cifar10(bsize):

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  #x_train = convert_to_grayscale(x_train)
  #x_test = convert_to_grayscale(x_test)
  print("CIFAR-10 loaded...")

  approx_mean = np.mean(x_train[:10000,:,:])
  print("Approx. Mean before normalization: " + str(approx_mean))

  #x_train = (x_train.astype(np.float32).reshape(-1, 32*32) - approx_mean ) / 255.0
  #x_test = (x_test.astype(np.float32).reshape(-1, 32*32) - approx_mean ) / 255.0

  x_train = (x_train.astype(np.float32) - approx_mean ) / 255.0
  x_test = (x_test.astype(np.float32) - approx_mean ) / 255.0

  napprox_mean = np.mean(x_train[10001:20000,:])
  print("Approx. Mean after normalization: " + str(napprox_mean))

  dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_train, tf.float32),
     tf.cast(y_train,tf.int64)))
  dataset = dataset.shuffle(100).batch(bsize)

  testdataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_test, tf.float32),
     tf.cast(y_test,tf.int64)))
  testdataset = testdataset.shuffle(100).batch(bsize)


  return dataset,testdataset

def get_cifar100():

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
  x_train = convert_to_grayscale(x_train)
  x_test = convert_to_grayscale(x_test)
  print("CIFAR-100 loaded...")

  approx_mean = np.mean(x_train[:10000,:,:])
  print("Approx. Mean before normalization: " + str(approx_mean))

  x_train = (x_train.astype(np.float32).reshape(-1, 32*32) - approx_mean ) / 255.0
  x_test = (x_test.astype(np.float32).reshape(-1, 32*32) - approx_mean ) / 255.0

  napprox_mean = np.mean(x_train[10001:20000,:])
  print("Approx. Mean after normalization: " + str(napprox_mean))

  dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_train, tf.float32),
     tf.cast(y_train,tf.int64)))
  dataset = dataset.shuffle(100).batch(1)

  testdataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_test, tf.float32),
     tf.cast(y_test,tf.int64)))
  testdataset = testdataset.shuffle(100).batch(1)

  return dataset,testdataset

def get_splice(x,i):
    return x[:,i]

class ELUModel(tf.keras.Model):
  def __init__(self, modeltype = 'mnist'):
    super(ELUModel, self).__init__()
    if modeltype == 'mnist':
      self.dense1 = tf.keras.layers.Dense(units=4096,activation='elu',input_shape=(1,768))
    else:
      self.dense1 = tf.keras.layers.Dense(units=4096,activation='elu',input_shape=(1,1024))

    self.dense2 = tf.keras.layers.Dense(units=4096,activation='elu',input_shape=(1,4096))
    #self.dense3 = tf.keras.layers.Dense(units=300,activation='elu',input_shape=(1,500))
    #self.dense4 = tf.keras.layers.Dense(units=100,activation='elu',input_shape=(1,300))    

    self.dense5 = tf.keras.layers.Dense(units=10,activation='elu',input_shape=(1,4096))


  def call(self, input,training=True):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    #result = self.dense3(result)
    #result = self.dense4(result)  # reuse variables from dense2 layer
    result = self.dense5(result)  # reuse variables from dense2 layer
    return result

class ConvModel(tf.keras.Model):
  def __init__(self, modeltype = 'mnist'):
    super(ConvModel, self).__init__()
    if modeltype == 'mnist':
      self.dense1 = tf.keras.layers.Dense(units=4096,activation='elu',input_shape=(1,768))
    else:
      self.dense1 = tf.keras.layers.Dense(units=4096,activation='elu',input_shape=(1,1024))

    self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(11, 11), strides=(1, 1),activation='relu',input_shape=(1,1024))
    self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    self.conv2 = tf.keras.layers.Conv2D(64, (7, 7), activation='relu')
    self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.conv3 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
    self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(1000, activation='relu')
    self.dense2 = tf.keras.layers.Dense(10, activation='relu')



  def call(self, input,training=True):
    """Run the model."""
    result = self.conv1(input)
    result = self.maxpool1(result)
    result = self.conv2(input)
    result = self.maxpool2(result)
    result = self.conv3(result)
    result = self.maxpool3(result)
    result = self.flatten(result)
    result = self.dense1(result)
    result = self.dense2(result)

    return result

class ConvDropoutModel(tf.keras.Model):
  def __init__(self, modeltype = 'mnist'):
    super(ConvDropoutModel, self).__init__()
    if modeltype == 'mnist':
      self.dense1 = tf.keras.layers.Dense(units=4096,activation='elu',input_shape=(1,768))
    else:
      self.dense1 = tf.keras.layers.Dense(units=4096,activation='elu',input_shape=(1,1024))

    self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(11, 11), strides=(1, 1),activation='relu',input_shape=(1,1024))
    self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    self.conv2 = tf.keras.layers.Conv2D(64, (7, 7), activation='relu')
    self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.conv3 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
    self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(1000, activation='relu')
    self.dropout = tf.keras.layers.Dropout(rate = .5)
    self.dense2 = tf.keras.layers.Dense(10, activation='relu')



  def call(self, input,training=True):
    """Run the model."""
    result = self.conv1(input)
    result = self.maxpool1(result)
    result = self.conv2(input)
    result = self.maxpool2(result)
    result = self.conv3(result)
    result = self.maxpool3(result)
    result = self.flatten(result)
    result = self.dense1(result)
    result = self.dropout(result,training)
    result = self.dense2(result)

    return result

class DropoutModel(tf.keras.Model):
  def __init__(self, modeltype = 'mnist'):
    super(DropoutModel, self).__init__()
    if modeltype == 'mnist':
      self.dense1 = tf.keras.layers.Dense(units=500,activation='elu',input_shape=(1,768))
    else:
      self.dense1 = tf.keras.layers.Dense(units=500,activation='elu',input_shape=(1,1024))

    self.dense2 = tf.keras.layers.Dense(units=500,activation='elu',input_shape=(1,500))
    self.dropout2 = tf.keras.layers.Dropout(rate=.35,input_shape=(1,500))

    self.dense3 = tf.keras.layers.Dense(units=300,activation='elu',input_shape=(1,500))
    self.dropout3 = tf.keras.layers.Dropout(rate=.35,input_shape=(1,300))

    self.dense4 = tf.keras.layers.Dense(units=100,activation='elu',input_shape=(1,300))    

    self.dense5 = tf.keras.layers.Dense(units=10,activation='elu',input_shape=(1,100))

  def call(self, input,training=True):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dropout2(result,training)
    result = self.dense3(result)  # reuse variables from dense2 layer
    result = self.dropout3(result,training)  # reuse variables from dense2 layer
    result = self.dense4(result)  # reuse variables from dense2 layer  
    result = self.dense5(result)      
    return result

class DropalanModel(tf.keras.Model):
  def __init__(self, modeltype = 'mnist'):
    super(DropalanModel, self).__init__()
    if modeltype == 'mnist':
      self.dense1 = tf.keras.layers.Dense(units=4096,activation='elu',input_shape=(1,768))
    else:
      self.dense1 = tf.keras.layers.Dense(units=4096,activation='elu',input_shape=(1,1024))

    self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(11, 11), strides=(1, 1),activation='relu',input_shape=(1,1024))
    self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    self.conv2 = tf.keras.layers.Conv2D(64, (7, 7), activation='relu')
    self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.conv3 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
    self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(1000, activation=None)
    self.dropalan = dropalan.Dropalan(rate = .5)
    self.dense2 = tf.keras.layers.Dense(10, activation='relu')

  def call(self, input,training=True):
    """Run the model."""
    result = self.conv1(input)
    result = self.maxpool1(result)
    result = self.conv2(input)
    result = self.maxpool2(result)
    result = self.conv3(result)
    result = self.maxpool3(result)
    result = self.flatten(result)
    result = self.dense1(result)
    result = self.dropalan(result,training)
    result = self.dense2(result)

    return result

mnist_model = DropalanModel(modeltype = 'cifar10')
dataset,testdataset = get_cifar10(16)


optimizer = tf.train.GradientDescentOptimizer(.01)
iteration = 0
total_loss = 0
display_update = 1000
validation_examples = 100
loss_history = []

for i in range(80):
    for (batch, (images, labels)) in enumerate(dataset.take(-1)):

      with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

      loss_history.append(loss_value.numpy())

      iteration += 1
      total_loss += loss_value.numpy()
      if iteration % display_update == 0:
        loss = total_loss/display_update
        print("Iteration: " + str(iteration) + ", loss: " + str(loss))
        total_loss = 0

        num_right = 0
        tot_num = 0
        for (tbatch, (timages, tlabels)) in enumerate(testdataset.take(validation_examples)):
            result_batch = mnist_model(timages,training=False).numpy()
            for b in range(result_batch.shape[0]):
                tot_num += 1
                if (np.argmax(result_batch[b]) == tlabels.numpy()[b]):
                    num_right += 1

        accuracy = num_right/tot_num
        print("Accuracy: " + str(accuracy))

        to_log = "iteration: " + str(iteration) + "\n" +"loss: " + str(loss) + "\n accuracy: " + str(accuracy) + "\n"

        with open("convdropalanprob2_cifar_noscale.txt", "a+") as myfile:
            myfile.write(to_log)

      grads = tape.gradient(loss_value, mnist_model.variables)
      optimizer.apply_gradients(zip(grads, mnist_model.variables),
                                global_step=tf.train.get_or_create_global_step())