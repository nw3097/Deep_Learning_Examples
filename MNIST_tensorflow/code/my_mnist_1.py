# goal: do softmax regression without using feed_dict

import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
FLAGS = None

def main(_):
  
  # Import data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = tf.data.Dataset.from_tensor_slices(mnist.train.images) # mnist.train.images is a numpy array
  train_labels = tf.data.Dataset.from_tensor_slices(np.asarray(mnist.train.labels, dtype=np.int32)).map(lambda z: tf.one_hot(z, 10))
  train_dataset = tf.data.Dataset.zip((train_data, train_labels)).shuffle(500).repeat().batch(100)

  eval_data = tf.data.Dataset.from_tensor_slices(mnist.test.images) # mnist.test.images is a numpy array
  eval_labels = tf.data.Dataset.from_tensor_slices(np.asarray(mnist.test.labels, dtype=np.int32)).map(lambda z: tf.one_hot(z, 10))
  eval_dataset = tf.data.Dataset.zip((eval_data, eval_labels)).shuffle(500).repeat().batch(100)

  # create general iterator
  iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
  next_element = iterator.get_next()

  # define initialization operations by dataset type
  training_init_op = iterator.make_initializer(train_dataset)
  eval_init_op = iterator.make_initializer(eval_dataset)

  # Create the model
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(next_element[0], W) + b

  # Define loss and optimizer
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=next_element[1], logits=y))
  
  # a train step involves forward prop: current weights are applied to current batch images to get current prediction,
  # then compare current prediction against current batch labels to compute loss, then backward prop where the derivatives are
  # calculated and weights are adjusted
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  # Train
  sess.run(training_init_op)
  for _ in range(1000):
    sess.run(train_step)

  # Test trained model
  sess.run(eval_init_op)
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(next_element[1], 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)