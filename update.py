import tensorflow as tf

dataset = [
        ([[0,0]], [0]),
        ([[1,0]], [1]),
        ([[0,1]], [1]),
        ([[1,1]], [0])]

# w = tf.Variable(tf.random_normal([2,1]))
# b = tf.Variable(tf.random_normal([1]))
w = tf.Variable(tf.zeros([2,1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape=[1, 2])
y = tf.placeholder(tf.float32, shape=[1])

pred = tf.sigmoid(tf.matmul(x, w) + b)

loss = tf.reduce_mean(tf.square(pred - y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(100):
        for x_ex, y_ex in dataset:
            sess.run(train_step, feed_dict={x: x_ex, y: y_ex})
            print(sess.run(w))

