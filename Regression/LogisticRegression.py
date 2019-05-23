import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()


def logistic_regression(features):
    hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis


def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.log(logistic_regression(features))
                           + (1 - labels) * tf.log(1 - hypothesis))
    return cost


def grad(features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features), features, labels)
    return tape.gradient(loss_value, [W, b])


def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy


score_train = [[3., 2.],
               [0., 1.],
               [1., 0.],
               [3., 4.],
               [4., 0.],
               [1., 2.]]

result_train = [[1.],
                [0.],
                [1.],
                [0.],
                [1.],
                [0.]]

score_test = [[2., 1.]]
result_test = [[1.]]

dataset = tf.data.Dataset.from_tensor_slices((score_train, result_train)).batch(len(score_train))

W = tf.Variable(tf.zeros([2, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

for step in range(1000):
    for score, result in tfe.Iterator(dataset):
        grads = grad(score, result)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 100 == 0:
            print(step, loss_fn(logistic_regression(score), score, result))

test_acc = accuracy_fn(logistic_regression(score_test), result_test)
print(test_acc)
