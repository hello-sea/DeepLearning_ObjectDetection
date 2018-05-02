# coding:utf-8

import os.path
import sys
import re
import os
import json

from mySelectivesearch import  mySelectivesearch

#python pkl 文件读写
import pickle as pickle

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class MyData():
    def __init__(self):
        self.data_filePath = []
        self.data_fileName = []
        # self.data_tpye = []

        self.data = []
        self.labels = []

# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    data = MyData()
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        if os.path.isfile(child):
            data.data_filePath.append(child)
            data.data_fileName.append(allDir)

    # # 显示
    # for i in array:
    #     print(i)      
    return data


def myFastGFile(py_data):
    # 新建一个Session
    with tf.Session() as sess:
        # path = py_data.data_filePath[0]
        for path in py_data.data_filePath:
            img, candidates = mySelectivesearch(path)
            for x, y, w, h in candidates:
                # print(x, y, w, h)
                img_data = img[y:y+h, x:x+w]
                # plt.imshow(img_data)
                # plt.show()

                data = tf.convert_to_tensor(img_data) # np 转换为 tensor
                # 改变图片尺寸            
                resized = tf.image.resize_images(data, [28, 28], method=0)
                # 设定 shape
                resized = tf.reshape(resized, [28, 28, 3]) #最后一维代表通道数目，如果是rgb则为3 
                # 标准化
                standardization_image = tf.image.per_image_standardization(resized)#标准化
                # 设定 shape
                resized = tf.reshape(standardization_image, [-1]) #最后一维代表通道数目，如果是rgb则为3 
                # resized = tf.reshape(resized, [-1]) #最后一维代表通道数目，如果是rgb则为3 
                # 转为 np类型
                py_data.data.append(resized.eval())
                py_data.labels.append( 31 - 1 )

        '''
        # #验证数据转换正确
        resized = tf.reshape(py_data.data[0], [28, 28, 3])
        resized = np.asarray(resized.eval(), dtype='uint8')        
        plt.imshow(resized)
        plt.show()
        '''

def saveData(py_data, filePath_data, filePath_labels):
    pass
    '''
    with tf.Session() as sess:
        train_data =tf.convert_to_tensor(np.array( trainData.data ) )
    '''
    data = np.array( py_data.data )
    labels = py_data.labels

    # import os
    if os.path.exists(filePath_data): #删除文件，可使用以下两种方法。
        os.remove(filePath_data)      #os.unlink(my_file)

    if os.path.exists(filePath_labels): #删除文件，可使用以下两种方法。
        os.remove(filePath_labels)      #os.unlink(my_file)

    with open(filePath_data,'wb') as f:
        pickle.dump(data, f)

    with open(filePath_labels,'wb') as f:
        pickle.dump(labels, f)

    print('\ndone!')

def run(filePath_loadData, filePath_data, filePath_labels):

    loadData = eachFile(filePath_loadData) #注意：末尾不加/
    myFastGFile(loadData)  
    saveData(loadData, filePath_data, filePath_labels)
    
    '''
    trainData = eachFile("../Data/logos/train") #注意：末尾不加/
    # for i in range(0,len(data.data_fileName)):
    #     print(data.data_tpye[i])
    #     print(data.data_oneHot_labels[i])

    myFastGFile(trainData)  
    saveData(trainData, 'Model/train_data.plk', 'Model/train_labels.plk')
    
    # print(trainData.data[0].shape)
    # print(trainData.data[0])

    '''


if __name__ == "__main__":
    print('目前系统的编码为：',sys.getdefaultencoding()) 


    run("../Data/background/car_images", 'Model/background_data.plk', 'Model/background_data_labels.plk')
