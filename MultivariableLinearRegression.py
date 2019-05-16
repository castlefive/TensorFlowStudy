import tensorflow as tf
import numpy as np
import math

tf.enable_eager_execution()


def predict(X):
    return tf.matmul(X, W) + b


avengers_data = np.array([
    [56., 14., 18., 42.],
    [94., 29., 39., 62.],
    [159., 35., 50., 66.]
], dtype=np.float32)

X = avengers_data[:, :-1]
y = avengers_data[:, [-1]]

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

learnging_rate = 0.00001

cost = 15
while int(cost) > 14:
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(tf.square(predict(X) - y))

    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learnging_rate * W_grad)
    b.assign_sub(learnging_rate * b_grad)

expecting_data = np.array([
    [214., 44., 66.]
], dtype=np.float32)

X = expecting_data[:, :]

print('Cost:', cost.numpy())
print('어벤져스4 예상 관객 수(UBD):', math.floor(predict(X)))
