# import json

# def myJsonLoad(filePath):
#     '''把文件打开从字符串转换成数据类型'''
#     with open(filePath,'rb') as load_file:
#         load_dict = json.load(load_file)
#         return load_dict

# logo_id =myJsonLoad('Data/logo_30/id_label.json') 
# print(logo_id)




# import re
# allDir =  'aodi.500_0aic_QXOX3HGW_0DHP431L.jpg'
# theTpye = re.split('\.',allDir )[0]
# print(theTpye)


# import tensorflow as tf
# def myOneHot(num_classes):
#     NUM_CLASSES = num_classes  # 分类个数

#     labels = list(range(0,num_classes,1)) # sample label

#     batch_size = tf.size(labels) # get size of labels : 4

#     labels = tf.expand_dims(labels, 1) # 增加一个维度
#     indices = tf.expand_dims(tf.range(0, batch_size,1), 1) #生成索引
#     concated = tf.concat([indices, labels] , 1) #作为拼接
#     onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0) # 生成one-hot编码的标签    
#     with tf.Session() as ssess:
#         # print(onehot_labels.eval())
#         return onehot_labels.eval()

# ontHot = myOneHot(30)
# print(ontHot[1])



# import tensorflow as tf
# x = tf.placeholder(tf.float32,[None, 784*3])
# data = tf.reshape(x, [-1, 28, 28, 3]) #最后一维代表通道数目，如果是rgb则为3 

# print(x)
# print(data)






# import tensorflow as tf
# data = tf.placeholder(tf.float32,[1, 5])
# # with tf.Session() as sess:
# print(data)




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
