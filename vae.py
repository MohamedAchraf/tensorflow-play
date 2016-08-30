import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('logdir', 'summaries/', 'The summaries folder')

cifar10.maybe_download_and_extract()

images, labels = cifar10.distorted_inputs()

# TODO(irapha): add max pooling

# First convolution
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 25], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[25]))

h_conv1 = tf.nn.relu(
        tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

# Second convolution, with downsizing
W_conv2 = tf.Variable(tf.truncated_normal([7, 7, 25, 5], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[5]))

h_conv2 = tf.nn.relu(
        tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 3, 3, 1], padding='SAME') + b_conv2)

# Fully connected layer, towards latent representation (of len 25)
h_conv2_vec = tf.reshape(h_conv2, [128, -1])
W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 5, 25], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[25]))

h_fc1 = tf.nn.relu(tf.matmul(h_conv2_vec, W_fc1) + b_fc1)

# Second fully connected layer, from latent to downsized image vector
W_fc2 = tf.Variable(tf.truncated_normal([25, 8 * 8 * 5], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[8 * 8 * 5]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# First deconvolution
h_fc2_mtrx = tf.reshape(h_fc2, [128, 8, 8, 5])
W_deconv1 = tf.Variable(tf.truncated_normal([7, 7, 25, 5], stddev=0.1))
b_deconv1 = tf.Variable(tf.constant(0.1, shape=[25]))

h_deconv1 = tf.nn.relu(
        tf.nn.conv2d_transpose(h_fc2_mtrx, W_deconv1, [128, 8, 8, 25], [1, 1, 1, 1], padding='SAME') + b_deconv1)

# Second deconvolution, outputs image.
W_deconv2 = tf.Variable(tf.truncated_normal([5, 5, 3, 25], stddev=0.1))
b_deconv2 = tf.Variable(tf.constant(0.1, shape=[3]))

h_deconv2 = tf.nn.relu(
        tf.nn.conv2d_transpose(h_deconv1, W_deconv2, [128, 24, 24, 3], [1, 3, 3, 1], padding='SAME') + b_deconv2)

# Loss
loss = tf.reduce_mean(tf.square(h_deconv2 - images))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

tf.scalar_summary('loss', loss)
summary_op = tf.merge_all_summaries()


# Training
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)

    for i in range(100):
        summary, _ = sess.run([summary_op, train_step])
        summary_writer.add_summary(summary, i)

    coord.request_stop()
    coord.join(threads)
