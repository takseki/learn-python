# coding: utf-8

# sparse auto encoder (tensorflow)

# 下記サイトのコードをベースにして、スパース正則化を追加
#   https://qiita.com/mokemokechicken/items/8216aaad36709b6f0b5c
# 
# 岡谷 "深層学習" を参考に全体を調整
#   隠れ層次元を100に
#   ドロップアウトを外し、中間層をRelu, 出力層を恒等関数に
#   重み、バイアス初期値を調整

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
H = 100
BATCH_SIZE = 100
DROP_OUT_RATE = 0.5


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


# Input: x : 28*28=784
x = tf.placeholder(tf.float32, [None, 784])

# Variable: W, b1
W = weight_variable((784, H))
b1 = bias_variable([H])

# Hidden Layer: h
#h = tf.nn.softsign(tf.matmul(x, W) + b1)
h = tf.nn.relu(tf.matmul(x, W) + b1)
keep_prob = tf.placeholder("float")
#h_drop = tf.nn.dropout(h, keep_prob)

# Variable: b2
W2 = tf.transpose(W)  # 転置
b2 = bias_variable([784])
#y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)
y = tf.matmul(h, W2) + b2

# 中間層に固定の値を入れた時の出力を観測するために追加
# Variable: yy
hh = tf.placeholder(tf.float32, [H])
#yy = tf.nn.relu(tf.tensordot(hh, W2, 1) + b2)
yy = tf.tensordot(hh, W2, 1) + b2

# Define Loss Function
reconstruct_loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

# スパース正則項
# todo: ミニバッチでのみ平均化しているが、本によれば、
#       ミニバッチ間の逐次平均が必要っぽい
rho = tf.constant(0.05)
#rho_hat = tf.reduce_mean(h_drop, 0)
rho_hat = tf.reduce_mean(h, 0)
#sparse_loss = tf.reduce_mean(rho * tf.log(tf.clip_by_value(rho, 1e-10, 1))
sparse_loss = tf.reduce_sum(rho * tf.log(tf.clip_by_value(rho, 1e-10, 1))
                             - rho * tf.log(tf.clip_by_value(rho_hat, 1e-10, 1))
                             + (1 - rho) * tf.log(tf.clip_by_value(1 - rho, 1e-10, 1))
                             - (1 - rho) * tf.log(tf.clip_by_value(1 - rho_hat, 1e-10, 1)))

beta = tf.constant(0.1)
loss = reconstruct_loss + beta * sparse_loss

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
        print(y.eval(session=sess, feed_dict={x: batch_xs, keep_prob: 1.0}))
        
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
        unit[i] = 5.0
        eigen = yy.eval(session=sess, feed_dict={hh: unit, keep_prob: 1.0})
        plt.subplot(N_ROW, N_COL, row*N_COL+col+1)
        plt.imshow(eigen.reshape((28, 28)), cmap="magma", clim=(0, 1.0), origin='upper')
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
plt.savefig("weight.png")
plt.show()
