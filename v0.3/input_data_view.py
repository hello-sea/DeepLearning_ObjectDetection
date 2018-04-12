

import tensorflow as tf
# python pkl 文件读写
import pickle as pickle

train_data_np = pickle.load(open('Model/train_data.plk', 'rb'))
train_labels = pickle.load(open('Model/train_labels.plk', 'rb'))

eval_data_np = pickle.load(open('Model/eval_data.plk', 'rb'))
eval_labels = pickle.load(open('Model/eval_labels.plk', 'rb'))

with tf.Session() as sess:
    train_data = tf.convert_to_tensor(train_data_np)
    eval_data = tf.convert_to_tensor(eval_data_np)

    print(train_data.eval())
# print(train_labels)
    print(eval_data.eval())
# print(eval_labels)