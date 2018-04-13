# coding:utf-8

import numpy as np
import tensorflow as tf
import pickle as pickle # python pkl 文件读写


if __name__ == "__mian__":
    eval_data = np.array(pickle.load(open('Model/eval_data.plk', 'rb')) )
    eval_labels = np.array(pickle.load(open('Model/eval_labels.plk', 'rb')) )

    saver = tf.train.Saver()  
    
    with tf.Session() as sess:  
        saver.restore(sess, "./Model/cnn_model.ckpt") # 注意此处路径前添加"./"  
        
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False) 
        eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

