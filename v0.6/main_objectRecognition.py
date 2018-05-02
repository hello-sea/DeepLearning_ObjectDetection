
import numpy as np
import tensorflow as tf
import pickle as pickle # python pkl 文件读写

from cnn_model import cnn_model_fn
from mySelectivesearch import  mySelectivesearch

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    path = '../Data/train_data/LabelData/500_0LmA_rVnydZ4z_CDcA8yqW.jpg'
    
    # path = '../Data/logo_30/logos/jeep.jpg'

    img, candidates = mySelectivesearch(path)
    predict_data = []
    with tf.Session() as sess:
        for x, y, w, h in candidates:
            img_data = img[y:y+h, x:x+w]
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

    for e in  predict_results:
        if e['probabilities'][e['classes']] > 0.3 :
            print(e['classes'])
            print(e['probabilities'])
    print("done!")


if __name__ == "__main__":
    
    tf.app.run()


