# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
H = 50
BATCH_SIZE = 100
DROP_OUT_RATE = 0.5


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Input: x : 28*28=784
x = tf.placeholder(tf.float32, [None, 784])

# Variable: W, b1
W = weight_variable((784, H))
b1 = bias_variable([H])

# Hidden Layer: h
# softsign(x) = x / (abs(x)+1); https://www.google.co.jp/search?q=x+%2F+(abs(x)%2B1)
h = tf.nn.softsign(tf.matmul(x, W) + b1)
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h, keep_prob)

# Variable: b2
W2 = tf.transpose(W)  # 転置
b2 = bias_variable([784])
y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

# 中間層に固定の値を入れた時の出力を観測するために追加
# Variable: yy
hh = tf.placeholder(tf.float32, [H])
yy = tf.nn.relu(tf.tensordot(hh, W2, 1) + b2)

# Define Loss Function
loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

# For tensorboard learning monitoring
tf.summary.scalar("l2_loss", loss)

# Use Adam Optimizer
train_step = tf.train.AdamOptimizer().minimize(loss)

# Prepare Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
summary_writer = tf.summary.FileWriter('summary/l2_loss', graph=sess.graph)

# Training
for step in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, keep_prob: (1-DROP_OUT_RATE)})
    # Collect Summary
    summary_op = tf.summary.merge_all()
    summary_str = sess.run(summary_op, feed_dict={x: batch_xs, keep_prob: 1.0})
    summary_writer.add_summary(summary_str, step)
    # Print Progress
    if step % 100 == 0:
        print(loss.eval(session=sess, feed_dict={x: batch_xs, keep_prob: 1.0}))
        
# Draw Encode/Decode Result
N_COL = 10
N_ROW = 2
plt.figure(figsize=(N_COL, N_ROW*2.5))
batch_xs, _ = mnist.train.next_batch(N_COL*N_ROW)
for row in range(N_ROW):
    for col in range(N_COL):
        i = row*N_COL + col
        data = batch_xs[i:i+1]
        
        # Draw Input Data(x)
        plt.subplot(2*N_ROW, N_COL, 2*row*N_COL+col+1)
        plt.title('IN:%02d' % i)
        plt.imshow(data.reshape((28, 28)), cmap="magma", clim=(0, 1.0), origin='upper')
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        
        # Draw Output Data(y)
        plt.subplot(2*N_ROW, N_COL, 2*row*N_COL + N_COL+col+1)
        plt.title('OUT:%02d' % i)
        y_value = y.eval(session=sess, feed_dict={x: data, keep_prob: 1.0})
        plt.imshow(y_value.reshape((28, 28)), cmap="magma", clim=(0, 1.0), origin='upper')
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        
plt.savefig("result.png")
plt.show()

# 中間層の各ユニットに相当する画像を出力
N_COL = 10
N_ROW = H // N_COL
plt.figure(figsize=(N_COL, N_ROW))
for row in range(N_ROW):
    for col in range(N_COL):
        i = row*N_COL + col
        unit = np.zeros(H)
        unit[i] = 1
        eigen = yy.eval(session=sess, feed_dict={hh: unit, keep_prob: 1.0})
        plt.subplot(N_ROW, N_COL, row*N_COL+col+1)
        plt.imshow(eigen.reshape((28, 28)), cmap="magma", clim=(0, 1.0), origin='upper')
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
plt.show()
