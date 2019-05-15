import tensorflow as tf
import math
import matplotlib.pyplot as plt

tf.enable_eager_execution()

avengers_series = [1, 2, 3]
attendance_in_UBD = [42, 62, 66]

W = tf.Variable(0.0)
b = tf.Variable(0.0)

learnging_rate = 0.01

for i in range(5001):
    with tf.GradientTape() as tape:
        hypothesis = W * avengers_series + b
        cost = tf.reduce_mean(tf.square(hypothesis - attendance_in_UBD))

    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learnging_rate * W_grad)
    b.assign_sub(learnging_rate * b_grad)

    if i % 1000 == 0:
        print(i, W.numpy(), b.numpy(), cost)

expected_audiences = float((W.numpy()) * 4 + float(b.numpy()))
print('어벤져스4 예상 관객 수(UBD):', math.floor(expected_audiences))

plt.title('Predicting Avengers4 Audience')
plt.xlabel('Avengers Series', labelpad=1)
plt.ylabel('Attendance(UBD)')

plt.plot([1, 4], [attendance_in_UBD[0], expected_audiences], color="r")
plt.xticks([1, 2, 3, 4])
plt.scatter(avengers_series, attendance_in_UBD, color="b")
plt.scatter(4, expected_audiences, color="b")

plt.show()
