
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import pickle as pickle # python pkl 文件读写

from cnn_model import cnn_model_fn
from mySelectivesearch import  mySelectivesearch

tf.logging.set_verbosity(tf.logging.INFO)


def myJsonLoad(filePath):
    '''把文件打开从字符串转换成数据类型'''
    with open(filePath,'rb') as load_file:
        load_dict = json.load(load_file)
        return load_dict


def main(unused_argv):

    
    id_dict =  myJsonLoad('../Data/logo_30/id_label_zh.json')
    for i in  id_dict:
        print(i)

    

if __name__ == "__main__":
    
    tf.app.run()


