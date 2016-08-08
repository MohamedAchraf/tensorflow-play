import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'summaries', 'The summaries folder')

NUM_POINTS = 2000
points = []
for i in range(NUM_POINTS):
    if np.random.random() > 0.5:
        points.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        points.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

# plot data
# df = pd.DataFrame({'x': [v[0] for v in points],
    # 'y': [v[1] for v in points]})
# sns.lmplot('x', 'y', data=df, fit_reg=False, size=6)
# plt.show()

# k = tf.placeholder(tf.int32, shape=(), name='k')
# TODO(irapha): make k a fed tensor. Requires changing the way centroids
# work. Probably making it a list of centroids instead.
k = 4
vectors = tf.placeholder(tf.float32, [None, 2], name='vectors')
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(
    tf.sub(expanded_vectors, expanded_centroids)), 2), 0)

means = tf.concat(0, [tf.reduce_mean(
    tf.gather(
        vectors,
        tf.reshape(
            tf.where(tf.equal(assignments, c)),
            [1, -1])),
    reduction_indices=[1]) for c in range(k)])

update_centroids = tf.assign(centroids, means)

tf.histogram_summary('centroids', centroids)

sess = tf.Session()
sess.run(tf.initialize_all_variables(), feed_dict={vectors: points, k: 4})

for step in range(100):
    _, centroid_values, assignment_values = sess.run(
            [update_centroids, centroids, assignments],
            feed_dict={vectors: points, k: 4})

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

# plot data prettily
data = {'x': [], 'y': [], 'cluster': []}

for i in range(len(assignment_values)):
    data['x'].append(points[i][0])
    data['y'].append(points[i][1])
    data['cluster'].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot('x', 'y', data=df, fit_reg=False, size=6, hue='cluster', legend=False)
plt.show()
