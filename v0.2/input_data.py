# coding:utf-8

import os.path
import sys
import re
import os
import json

#python pkl 文件读写
import pickle as pickle

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def myJsonLoad(filePath):
    '''把文件打开从字符串转换成数据类型'''
    with open(filePath,'rb') as load_file:
        load_dict = json.load(load_file)
        return load_dict

def load_id_zh():
    return myJsonLoad('../Data/logo_30/id_label_zh.json') 
def load_id_us():
    return myJsonLoad('../Data/logo_30/id_label_us.json') 

class MyData():
    def __init__(self):
        self.data_filePath = []
        self.data_fileName = []
        self.data_tpye = []

        self.data = []
        self.labels = []

# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    data = MyData()
    id_dict = load_id_us()
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        if os.path.isfile(child):
            data.data_filePath.append(child)
            data.data_fileName.append(allDir)
            theTpye = re.split('\.',allDir)[0]
            # print(theTpye)
            data.data_tpye.append( theTpye )
            data.labels.append( int(id_dict[theTpye]) -1 )
    # # 显示
    # for i in array:
    #     print(i)      
    return data


def myFastGFile(py_data):
    # 新建一个Session
    with tf.Session() as sess:
        '''
        image_raw_data = tf.gfile.FastGFile(py_data.data_filePath[0], 'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data)
        plt.imshow(img_data.eval())
        plt.show()

        resized = tf.image.resize_images(img_data, [28, 28], method=0)
        print(resized)
        resized = tf.reshape(resized, [28, 28, 3]) #最后一维代表通道数目，如果是rgb则为3 
        print(resized)
        # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
        print("Digital type: ", resized.dtype)
        resized = np.asarray(resized.eval(), dtype='uint8')
        
        # tf.image.convert_image_dtype(rgb_image, tf.float32)
        plt.imshow(resized)
        plt.show()
        '''
        # path = py_data.data_filePath[0]
        for path in py_data.data_filePath:
            # 读取文件
            image_raw_data = tf.gfile.FastGFile(path, 'rb').read()
            # 解码
            img_data = tf.image.decode_jpeg(image_raw_data)
            # print(img_data)
            # 转灰度图
            img_data = sess.run(tf.image.rgb_to_grayscale(img_data))  
            # 改变图片尺寸
            resized = tf.image.resize_images(img_data, [28, 28], method=0)
            # 设定 shape
            resized = tf.reshape(resized, [28, 28, 1]) #最后一维代表通道数目，如果是rgb则为3 
            # 标准化
            standardization_image = tf.image.per_image_standardization(resized)#标准化
            # print(standardization_image)
            # print(standardization_image.eval())
            resized = tf.reshape(standardization_image, [-1]) #最后一维代表通道数目，如果是rgb则为3 

            ## 链接     
            ## resized = tf.expand_dims(resized, 0) # 增加一个维度
            ## print(resized)
            ## print(py_data.data)
            ## test_data = tf.concat(0, [test_data, resized])
            
            py_data.data.append(resized.eval())

        '''
        # #验证数据转换正确
        resized = tf.reshape(py_data.data[0], [28, 28, 3])
        resized = np.asarray(resized.eval(), dtype='uint8')        
        plt.imshow(resized)
        plt.show()
        '''


if __name__ == "__main__":
    print('目前系统的编码为：',sys.getdefaultencoding()) 


    trainData = eachFile("../Data/logos_test") #注意：末尾不加/
    # for i in range(0,len(data.data_fileName)):
    #     print(data.data_tpye[i])
    #     print(data.data_oneHot_labels[i])

    myFastGFile(trainData)  
    # print(trainData.data[0].shape)
    # print(trainData.data[0])
    '''
    with tf.Session() as sess:
        train_data =tf.convert_to_tensor(np.array( trainData.data ) )
    '''
    train_data = np.array( trainData.data )
    train_labels = trainData.labels

    # import os
    if os.path.exists('Model/train_data.plk'): #删除文件，可使用以下两种方法。
        os.remove('Model/train_data.plk')      #os.unlink(my_file)
    if os.path.exists('Model/train_labels.plk'): #删除文件，可使用以下两种方法。
        os.remove('Model/train_labels.plk')      #os.unlink(my_file)


    with open('Model/train_data.plk','wb') as f:
        pickle.dump(train_data, f)
    with open('Model/train_labels.plk','wb') as f:
        pickle.dump(train_labels, f)

    print('done!')

    ''' ****************************************************************************************** '''

    # evalData = eachFile("../Data/logos/eval") #注意：末尾不加/
    # # for i in range(0,len(data.data_fileName)):
    # #     print(data.data_tpye[i])
    # #     print(data.data_oneHot_labels[i])

    # myFastGFile(evalData)  
    # # print(evalData.data[0].shape)
    # # print(evalData.data[0])
    # '''
    # with tf.Session() as sess:
    #     train_data =tf.convert_to_tensor(np.array( evalData.data ) )
    # '''
    # eval_data = np.array( evalData.data )
    # eval_labels = evalData.labels

    # if os.path.exists('Model/eval_data.plk'): #删除文件，可使用以下两种方法。
    #     os.remove('Model/eval_data.plk')      #os.unlink(my_file)
    # if os.path.exists('Model/eval_labels.plk'): #删除文件，可使用以下两种方法。
    #     os.remove('Model/eval_labels.plk')      #os.unlink(my_file)


    # with open('Model/eval_data.plk','wb') as f:
    #     pickle.dump(eval_data, f)
    # with open('Model/eval_labels.plk','wb') as f:
    #     pickle.dump(eval_labels, f)

    # print('done!')


