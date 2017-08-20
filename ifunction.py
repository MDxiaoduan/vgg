import numpy as np


def trans_mul_to_one(vec):  # vec[kernel_size, kernel_size, output_channel] channel表示第几个
    kernel_size, _, output_channel = vec.shape
    # print(kernel_size, kernel_size, input_channel, output_channel)
    size = np.sqrt(output_channel)
    if size**2 == output_channel:
        weight = int(size)
        height = int(size)
    else:
        weight = int(size) + 1
        height = int(size)
    out_vec = np.zeros((height*kernel_size, weight*kernel_size))
    # print(out_vec.shape)
    for ii in range(height):
        for kk in range(weight):
            # print(ii,kk,height,weight)
            if ii*weight + kk < output_channel:
                out_vec[ii * kernel_size:(ii + 1) * kernel_size, kk * kernel_size:(kk + 1) * kernel_size] = \
                    vec[:, :, ii * weight + kk]
            else:
                return out_vec
    return out_vec

#  归一化
def Norm(vec):  # 默认变量值None!!
    if np.amax(vec) == np.min(vec):
        if np.amax(vec) == 0:
            return vec
        else:
            return vec / np.amax(vec)
    else:
        return (vec - np.min(vec)) / (np.amax(vec) - np.min(vec))


def Bina(im):                                    # 图像二值化
    h = im.shape[0]  # h是高
    w = im.shape[1]  # w是宽
    im_out = im.copy() # 这里不用copy的话会后面直接改变im1的内容
    for k in range(h):
        for j in range(w):
            if im_out[k, j] < 125:        # 阈值125
                im_out[k, j] = 0
            else:
                im_out[k, j] = 255
    return im_out


def one_hot(batch_label, class_num):   # label 是batch  class_num 是分类数  输出one_hot的label
    length = batch_label.shape[0]
    Class_list = [kk for kk in range(class_num)]
    out_label = np.zeros((length, class_num))
    for kk in range(length):
        out_label[kk, Class_list.index(int(batch_label[kk]))] = 1.0
    return out_label


def learning_rate(kk, name):
    if name is "exp":
        return 0.0001 + (1 - 0.0001) * np.exp(-kk / 100)
    elif name is "my":
        if kk < 100:
            return 1
        elif 100 < kk < 500:
            return 0.1
        elif 500 < kk < 1000:
            return 0.01
        elif 1000 < kk < 5000:
            return 0.001
        else:
            return 0.0001
    elif name is "0.01":
        return 0.01
    elif name is "0.001":
        return 0.001
    elif name is "0.0001":
        return 0.0001

