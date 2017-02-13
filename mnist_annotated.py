from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data', 'Directory for storing input data')
flags.DEFINE_string('train_dir', 'summaries', 'Directory for storing summaries')
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# Placeholders
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_truth = tf.placeholder(tf.float32, [None, 10], name='y_truth')

# First layer
W1 = tf.Variable(tf.truncated_normal([784, 20]), name='w1')
b1 = tf.Variable(tf.truncated_normal([20]), name='b1')
z = tf.matmul(x, W1) + b1
h = tf.nn.sigmoid(z)

tf.histogram_summary('W1', W1)

# Second layer
W2 = tf.Variable(tf.truncated_normal([20, 10]), name='w2')
b2 = tf.Variable(tf.truncated_normal([10]), name='b2')
y = tf.matmul(h, W2) + b2

tf.histogram_summary('W2', W2)

# Define loss and optimizer
cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_loss)

tf.scalar_summary('cross_entropy_loss', cross_entropy_loss)

# Start session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

# Train
for step in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary, _ = sess.run([summary_op, train_step], feed_dict={x: batch_xs, y_truth: batch_ys})
    summary_writer.add_summary(summary, step)

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_truth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
    y_truth: mnist.test.labels}))
