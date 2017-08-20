import tensorflow as tf
import numpy as np
from ifunction import one_hot
import _init_

class read_tf:
    def __init__(self):
        self.batch_size = _init_.batch_size
        self.weight = _init_.input_image[0]
        self.height = _init_.input_image[1]
        self.tfrecode_cwd = 'D:\\ipython\\data\\ImageNet\\read\\data\\tfrecode\\Imagenet_10_48.tfrecords'
        self.class_num = _init_.classes_numbers

    def read_and_decode(self):   # 读入tfrecords
        filename_queue = tf.train.string_input_producer([self.tfrecode_cwd])  # 生成一个queue队列
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })  # 将image数据和label取出来

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [self.height, self.weight, 3])  # reshape为128*128的3通道图片
        label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
        # 如果要输出batch

        # # 使用shuffle_batch可以随机打乱输入
        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                        batch_size=self.batch_size, capacity=13001,
                                                        min_after_dequeue=13000)
        """
         capacity: An integer. The maximum number of elements in the queue.
         min_after_dequeue: Minimum number elements in the queue after a
                     dequeue, used to ensure a level of mixing of elements.
        """
        # image_batch, label_batch = tf.train.batch([img_batch, label_batch],
        #                                             batch_size=batch_size,
        #                                             num_threads=64,
        #                                             capacity=3200)
        # img_batch = tf.cast(img_batch, tf.float32) * (1. / 255) - 0.5  # 归一化，在流中抛出img张量
        label_batch = tf.one_hot(label_batch, self.class_num)
        return img_batch, label_batch  # img, label

data = read_tf()
# img_batch, label_batch = data.read_and_decode()  # [25,224,224,3] label是0-9的数


def get_data():
    img_batch, label_batch = data.read_and_decode()  # [25,224,224,3] label是0-9的数
    with tf.Session() as sess:  # 开始一个会话
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)  # 启动所有的QueueRunners
        image, label = sess.run([img_batch, label_batch])
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / std
        coord.request_stop()
        coord.join(threads)
        sess.close()
        return image, label


class get_images:
    def __init__(self):
        self.data = np.load(_init_.data_path)
        self.batch_size = _init_.batch_size
        self.count = 0
        self.length = self.data.shape[0]
        self.weight = _init_.input_image[0]
        self.height = _init_.input_image[1]
        self.data_images = np.zeros((self.batch_size, self.weight, self.height, 3))
        self.class_num = _init_.classes_numbers
        self.data_label = np.zeros(self.batch_size)

    def get_mini_batch(self):
        if (self.count+1)*self.batch_size < self.length:
            images, label = self.call_data()
            self.count += 1
            return images, label
        else:
            self.count = 0
            return self.call_data()

    def call_data(self):
        for kk in range(self.batch_size):
            self.data_images[kk, :, :, :] = np.reshape(
                self.data[self.count * self.batch_size + kk, :self.weight * self.height * 3],
                (self.height, self.weight, 3))
            self.data_label[kk] = self.data[self.count * self.batch_size + kk, self.weight * self.height * 3:]
        mean = np.mean(self.data_images)
        std = np.std(self.data_images)
        image_norm = (self.data_images - mean) / std
        label_one_hot = one_hot(self.data_label, self.class_num)
        return image_norm, label_one_hot
# def plot_images(images, labels):
#     for i in np.arange(0, data.batch_size):
#         plt.subplot(5, 5, i + 1)
#         plt.axis('off')
#         plt.title(labels[i], fontsize=14)
#         plt.subplots_adjust(top=1.5)
#         plt.imshow(images[i, :, :, :])
#     plt.show()

# for kk in range(10):
#     i, l = get_data()
#     plot_images(i, l)
#
# with tf.Session() as sess:  # 开始一个会话
#     sess.run(tf.global_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)   # 启动所有的QueueRunners
#     image, label = sess.run([img_batch, label_batch])
#     print(image.shape)
#     plot_images(image, label)
#     coord.request_stop()
#     coord.join(threads)
