from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
from tensorflow.python.util.tf_export import tf_export
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

class Dropalan(keras_layers.Dropout, base.Layer):

    def __init__(self, rate, noise_shape=None, seed=None, name=None, **kwargs):
        super(Dropalan, self).__init__(rate=rate,noise_shape=noise_shape,seed=seed,name=name,**kwargs)
        if self.seed is None:
          self.seed = np.random.randint(10e6)

    def call(self, inputs, training=None):

        #prob1
        #rate_matrix = inputs*(-1.2)
        #rate_matrix = tf.clip_by_value(rate_matrix,0,.8)

        #prob2
        rate_matrix=tf.clip_by_value(1.0+tf.divide(.2,inputs),.8,.95)*tf.sign(tf.nn.relu(-1*inputs))

        keep_prob = 1.0 - rate_matrix*1.0
        return K.in_train_phase(mydropout(inputs, keep_prob, seed=None), inputs,training=training)


def mydropout(x, keep_prob, seed=None,name=None):  # pylint: disable=invalid-name

  with tf.name_scope(name, "mydropout", [x]) as name:
    x = tf.convert_to_tensor(x, name="x")

    #keep_prob is a tensor of probabilities between 0.0 and 1.0
    keep_prob = tf.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")

    random_tensor = keep_prob
    random_tensor += tf.random.uniform(keep_prob.shape, seed=seed, dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)

    #ret = tf.multiply(tf.div(x, keep_prob),binary_tensor)
    ret = tf.multiply(x,binary_tensor)
    return ret