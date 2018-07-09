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


# Input
#  x : shape=(n_batch, n_pixel), n_pixel=28*28=784
x = tf.placeholder(tf.float32, [None, 784])

# Variable
#  W  : shape=(n_pixel, n_hidden)
#  b1 : shape=(n_hidden)
W = weight_variable((784, H))
b1 = bias_variable([H])

# Hidden Layer
#  h  : shape=(n_batch, n_hidden)
h = tf.nn.relu(tf.matmul(x, W) + b1)
keep_prob = tf.placeholder("float")  # drop-out用, 使っていない

# Variable
#  W2 : shape=(n_hidden, n_pixels)
#  b2 : shape=(n_pixels)
W2 = tf.transpose(W)  # 転置
b2 = bias_variable([784])

# Output
#  y  : shape=(n_batch, n_pixels)
y = tf.matmul(h, W2) + b2

# 2乗和誤差
#  サンプル数方向で和をとり、ミニバッチ内で平均化
reconstruct_loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

# スパース正則項
# todo: ミニバッチ内でのみ平均化しているが、本によれば、
#       ミニバッチ間も逐次平均が必要っぽい
rho = tf.constant(0.05)              # 活性化率の目標値
rho_hat = tf.reduce_mean(h, axis=0)  # 活性化率

# 隠れ層のユニット方向で和を取る
# 本に載っていた式に合わせてmeanではなくsumにしている
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
#        print(y.eval(session=sess, feed_dict={x: batch_xs, keep_prob: 1.0}))
        
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
eigen = W2.eval(session=sess) * 5.0  # 適当にスケーリング
for row in range(N_ROW):
    for col in range(N_COL):
        i = row*N_COL + col
        plt.subplot(N_ROW, N_COL, row*N_COL+col+1)
        plt.imshow(eigen[i].reshape((28, 28)), cmap="magma", clim=(0, 1.0), origin='upper')
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
plt.savefig("weight.png")
plt.show()
