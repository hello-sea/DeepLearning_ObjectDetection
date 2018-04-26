# coding:utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    path = '../Data/train_LabelData/LabelData/500_0LmA_rVnydZ4z_CDcA8yqW.jpg'
    
    with tf.Session() as sess:
        image_raw_data = tf.gfile.FastGFile(path, 'rb').read()
        
        img_data = tf.image.decode_jpeg(image_raw_data)
        print(type(img_data))
        # resized = tf.image.resize_images(img_data, [28, 28], method=0)




