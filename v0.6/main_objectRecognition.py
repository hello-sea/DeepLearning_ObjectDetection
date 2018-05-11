
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
import pickle as pickle # python pkl 文件读写

from cnn_model import cnn_model_fn
from mySelectivesearch import  mySelectivesearch, showImg

tf.logging.set_verbosity(tf.logging.INFO)


def myJsonLoad(filePath):
    '''把文件打开从字符串转换成数据类型'''
    with open(filePath,'rb') as load_file:
        load_dict = json.load(load_file)
        return load_dict


def main(unused_argv):

    path = '../Data/data/LabelData/'
    fileName = '500_2i6k_xqda503n_gm4NXsz9.jpg' #注意：含文件类型
    filePath = path + fileName
    # print(filePath)
    
    data_dict = myJsonLoad('../Data/data/data.json')
    id_dict =  myJsonLoad('../Data/logo_30/id_label_zh.json')

    candidates = []
    for i in data_dict:
        # print(i['image_id'])
        if i['image_id'] == fileName:
            for j in i['items']:
                # print(j['bbox'])
                candidates.append(j['bbox'])

    img, candidates_buf = mySelectivesearch(filePath)
    showImg(img, candidates_buf)

    predict_data = []
    with tf.Session() as sess:
        for x1, y1, x2, y2 in candidates:
            img_data = img[y1:y2, x1:x2]
            data = tf.convert_to_tensor(img_data) # np 转换为 tensor
            # 改变图片尺寸            
            resized = tf.image.resize_images(data, [28, 28], method=0)
            # 设定 shape
            resized = tf.reshape(resized, [28, 28, 3]) #最后一维代表通道数目，如果是rgb则为3 
            # 标准化
            standardization_image = tf.image.per_image_standardization(resized)#标准化
            # 设定 shape
            resized = tf.reshape(standardization_image, [-1]) #最后一维代表通道数目，如果是rgb则为3 
            # 转为 np类型
            predict_data.append(resized.eval())
    predict_data = np.array( predict_data )
    print(predict_data.shape)


    # Create the Estimator
    cnn_classifier = tf.estimator.Estimator(
        # model_fn=cnn_model_fn, model_dir="cnn_convnet_model")
        model_fn=cnn_model_fn, model_dir="Model")

    # predict
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        num_epochs=1,
        shuffle=False) 
    predict_results = cnn_classifier.predict(input_fn=predict_input_fn)
    print(type(predict_results))

    typeID = []
    typeName = []
    softmax = []
    for e in  predict_results:
        # if e['probabilities'][e['classes']] > 0.3 :
        number = e['classes']+1
        name = id_dict[ str( e['classes']+1 ) ]  
        typeID.append(number)
        typeName.append(name)
        print(number) #注意：json的分类是从1开始的
        print(name )
        print(e['probabilities'][number-1]) #为此类型的概率
        print(e['probabilities'])
        softmax.append(e['probabilities'] )
    print("done!")

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)

    tag = 0
    for x1, y1, x2, y2 in candidates:
        
        print(x1, y1, x2, y2)
        rect = mpatches.Rectangle(
            (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=1)
        # ax.text(x1,y1, typeName[tag] )   #.text(x, y, s, fontdict=None, withdash=False, **kwargs)
        tag = tag + 1  
        ax.add_patch(rect)
        
    plt.show()
    

if __name__ == "__main__":
    
    tf.app.run()


