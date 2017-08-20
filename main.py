import tensorflow as tf
from vgg_16_network import vgg16
from get_images import get_images
from ifunction import learning_rate
import _init_
import matplotlib.pyplot as plt
import numpy as np
vgg16 = vgg16()

data = get_images()

x = tf.placeholder(tf.float32, [None, 84, 84, 3])
y = tf.placeholder(tf.float32, [None, 10])

fc_out = vgg16.vgg(x, scope='vgg')
train_step, acc = vgg16.train_loss(fc_out, y)

saver = tf.train.Saver()
name = ["exp", "my", "0.01", "0.001", "0.0001"]

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True

accs ={"exp":[], "my":[], "0.01":[], "0.001":[], "0.0001":[]}
for n in name:
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for kk in range(_init_.iteration_numbers):
            img_batch, label_batch = data.get_mini_batch()
            vgg16.learning_rate = learning_rate(kk, n)
            step, accuracy = sess.run([train_step, acc], feed_dict={x: img_batch, y: label_batch})
            if kk % _init_.display_step == 0:
                accs[n].append(accuracy)
                print("Batch: ", kk, "Accuracy: ", accuracy)
        # acc_ = 0
        # for _ in range(100):
        #     img_batch, label_batch = data.get_mini_batch()
        #     accuracy = sess.run(acc, feed_dict={x: img_batch, y: label_batch})
        #     acc_ += accuracy
        # print("Accuracy: ", acc_/100)
        sess.close()

x = np.arange(len(accs["exp"]))
#  plt.plot(x, accs["exp"], "cs", x, accs["my"], "cs", x, accs["0.01"], "cs", x, accs["0.001"], "cs", x, accs["0.0001"], "cs")
line1, = plt.plot(x, accs["exp"], "cs", markersize=10)
line2, = plt.plot(x, accs["my"], "ro", markersize=10)
line3, = plt.plot(x, accs["0.01"], "g--", markersize=10)
line4, = plt.plot(x, accs["0.001"], "b+", markersize=10)
line5, = plt.plot(x, accs["0.0001"],  '--', color='black')
# Create a legend for the first line.
plt.legend([line1, line2, line3, line4, line5], ["lr--exp", "lr--Step_decay", "lr--0.01", "lr--0.001", "lr--0.0001"])
plt.xlabel('iter')
plt.ylabel('acc')
plt.savefig("examples.jpg")
plt.show()