import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('logdir', 'summaries', 'The summaries folder')
flags.DEFINE_string('exp', 'exp', 'The experiment name, used for logging.')
flags.DEFINE_boolean('train', False, 'Whether to perform training op.'
        'Tries restoring from checkpoint if false')
flags.DEFINE_integer('train_steps', 1000, 'How many steps tro train the vae for')

cifar10.maybe_download_and_extract()

images, labels = cifar10.distorted_inputs()

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

# TODO(irapha): make latent space larger, probably will help results
# TODO(irapha): make variational

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
# TODO(irapha): use cross entropy
loss = tf.reduce_mean(tf.square(h_deconv2 - images))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

tf.scalar_summary('loss', loss)

# Sampling from a given latent vector
tf.image_summary('initial_images', images, max_images=10)

h_conv1_sample = tf.nn.relu(
        tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_conv2_sample = tf.nn.relu(
        tf.nn.conv2d(h_conv1_sample, W_conv2, strides=[1, 3, 3, 1], padding='SAME') + b_conv2)
h_conv2_vec_sample = tf.reshape(h_conv2_sample, [128, -1])
h_fc1_sample = tf.nn.relu(tf.matmul(h_conv2_vec_sample, W_fc1) + b_fc1)
h_fc2_sample = tf.nn.relu(tf.matmul(h_fc1_sample, W_fc2) + b_fc2)
h_fc2_mtrx_sample = tf.reshape(h_fc2_sample, [128, 8, 8, 5])
h_deconv1_sample = tf.nn.relu(
        tf.nn.conv2d_transpose(h_fc2_mtrx_sample, W_deconv1, [128, 8, 8, 25], [1, 1, 1, 1], padding='SAME') + b_deconv1)
h_deconv2_sample = tf.nn.relu(
        tf.nn.conv2d_transpose(h_deconv1_sample, W_deconv2, [128, 24, 24, 3], [1, 3, 3, 1], padding='SAME') + b_deconv2)

tf.image_summary('final_images', h_deconv2_sample, max_images=10)

summary_op = tf.merge_all_summaries()

# Training
with tf.Session() as sess:
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    summary_writer = tf.train.SummaryWriter('%s/%s' % (FLAGS.logdir, FLAGS.exp), sess.graph)

    if FLAGS.train:
        # train op
        for i in range(FLAGS.train_steps):
            summary, _ = sess.run([summary_op, train_step])
            summary_writer.add_summary(summary, i)

        # save op
        saver.save(sess, '/tmp/vae.ckpt')
    else:
        saver.restore(sess, '/tmp/vae.ckpt')
        # sample op
        summary, _ = sess.run([summary_op, h_deconv2_sample])
        summary_writer.add_summary(summary, 0)

    coord.request_stop()
    coord.join(threads)
