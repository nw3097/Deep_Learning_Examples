# relative to my_mnist_7.py: replace softmax regression with convolutional neural network

import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

FLAGS = None
tf.summary.FileWriterCache.clear()

def weight_variable(shape):
  '''use this function to initialize weights of given shape'''
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  '''use this function to initialize biases of given shape'''
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  '''
  x: input tensor (batch_size, height, width, input_channels)
  W: filter tensor (filter_height, filter_width, input_channels, num_of_filters)
  output: (batch_size, output_height, output_width, num_of_filters)
  '''
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  '''
  x: input tensor (batch_size, height, width, input_channels)
  output: (batch_size, output_height, output_width, input_channels)
  '''
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
  
  # Import data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = tf.data.Dataset.from_tensor_slices(mnist.train.images) # mnist.train.images is np.array
  train_labels = tf.data.Dataset.from_tensor_slices(np.asarray(mnist.train.labels, dtype=np.int32)).map(lambda z: tf.one_hot(z, 10))
  train_dataset = tf.data.Dataset.zip((train_data, train_labels)).repeat().batch(100)

  eval_data = tf.data.Dataset.from_tensor_slices(mnist.test.images) 
  eval_labels = tf.data.Dataset.from_tensor_slices(np.asarray(mnist.test.labels, dtype=np.int32)).map(lambda z: tf.one_hot(z, 10))
  eval_dataset = tf.data.Dataset.zip((eval_data, eval_labels)).repeat().batch(100)

  # create general iterator
  iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
  next_element = iterator.get_next()

  # define initialization operations by dataset type
  training_init_op = iterator.make_initializer(train_dataset)
  eval_init_op = iterator.make_initializer(eval_dataset)

  W_conv1 = weight_variable([5, 5, 1, 32]) # weights of first conv layer
  b_conv1 = bias_variable([32]) # biases of first conv layer
  x_image_summary = tf.summary.image('input', tf.reshape(next_element[0], [-1, 28, 28, 1]), 3)
  x_image = tf.reshape(next_element[0], [-1,28,28,1]) # reshape x into 2d images
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # first layer convolution followed by relu activation
  h_pool1 = max_pool_2x2(h_conv1) # first max pool
  W_conv2 = weight_variable([5, 5, 32, 64]) # weights of second conv layer
  b_conv2 = bias_variable([64]) # biases of second conv layer
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # second layer convolution followed by relu activation
  h_pool2 = max_pool_2x2(h_conv2) # second max pool
  W_fc1 = weight_variable([7 * 7 * 64, 1024]) # weights of first fully connected layer
  b_fc1 = bias_variable([1024]) # biases of first fully connected layer
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # flatten the output from second max pool
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # output of first fully connected layer
  keep_prob = tf.placeholder(tf.float32) # dropout probability
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # modify output of first fully connected layer by dropout
  W_fc2 = weight_variable([1024, 10]) # weights of second fully connected layer
  b_fc2 = bias_variable([10]) # biases of second fully connected layer
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # output of second fully connected layer
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=next_element[1], logits=y_conv)) # loss operation
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # train operation  
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(next_element[1], 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  accuracy_scalar = tf.summary.scalar("accuracy",accuracy)

  # Add ops to save and restore all the variables.
  home = os.getenv('HOME')
  save_path = home + '/Deep_Learning_Examples/MNIST_tensorflow/checkpoints/'
  model_name = 'my_model'
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  saver = tf.train.Saver()
  save_path_full = os.path.join(save_path, model_name)

  # let the session begin
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # restore the session
  # new_saver = tf.train.import_meta_graph(save_path + '/my_model-1000.meta')
  # new_saver.restore(sess, tf.train.latest_checkpoint(save_path))

  # define writers
  summaries_dir = home + '/Deep_Learning_Examples/MNIST_tensorflow/board'
  train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(summaries_dir + '/test')

  # Train
  sess.run(training_init_op)
  for step in range(1000):
    sess.run(train_step, feed_dict={keep_prob: 0.5})
    if step % 20 == 0:
      sum1 = sess.run(x_image_summary)
      sum2 = sess.run(accuracy_scalar, feed_dict={keep_prob: 1})
      train_writer.add_summary(sum1,step)
      train_writer.add_summary(sum2,step)

    if (step+1) % 200 == 0:
      save_path = saver.save(sess, save_path_full, step+1)
      print("Model {} saved in path {}".format(step+1, save_path_full))

  # Test trained model
  sess.run(eval_init_op)
  sum3, validation_accuracy = sess.run([accuracy_scalar, accuracy], feed_dict={keep_prob: 1})
  test_writer.add_summary(sum3)
  print(validation_accuracy)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)