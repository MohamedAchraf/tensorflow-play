import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
training_data = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03) # OUR BIAS AND WEIGHT.
    training_data.append([x1, y1])

x_data, y_data = zip(*training_data)

plt.plot(x_data, y_data, 'ro')
plt.show()

# start TensorFlow
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'summaries', 'The summaries folder')

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')
x = tf.placeholder(tf.float32, shape=(None,), name='x')

with tf.name_scope('logit') as scope:
    y = W * x + b

with tf.name_scope('MSE') as scope:
    y_truth = tf.placeholder(tf.float32, shape=(None,), name='y_truth')
    loss = tf.reduce_mean(tf.square(y - y_truth))

tf.scalar_summary(loss.op.name, loss)

optimizer = tf.train.GradientDescentOptimizer(0.5)
gradients, variables = optimizer.compute_gradients(loss)

tf.histogram_summary('gradients', gradients)

train = optimizer.apply_gradients([gradients, variables])

# actually run things
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in range(15):
        summary, _ = sess.run([summary_op, train],
                feed_dict={x: x_data, y_truth: y_data})
        summary_writer.add_summary(summary, step)

    print(sess.run(W), sess.run(b))

    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.show()
