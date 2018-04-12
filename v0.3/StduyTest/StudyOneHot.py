import tensorflow as tf
# import os
# os.system('cls')

NUM_CLASSES = 30 # 10分类

labels = list(range(0,30,1)) # sample label
print("labels",labels)

batch_size = tf.size(labels) # get size of labels : 4

labels = tf.expand_dims(labels, 1) # 增加一个维度

indices = tf.expand_dims(tf.range(0, batch_size,1), 1) #生成索引
concated = tf.concat([indices, labels] , 1) #作为拼接
onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0) # 生成one-hot编码的标签

with tf.Session() as ssess:
    print("labels",labels.eval())
    print("indices",indices.eval())
    print("concated",concated.eval())
    print(onehot_labels[0])

