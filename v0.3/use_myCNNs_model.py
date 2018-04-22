# coding:utf-8

import numpy as np
import tensorflow as tf
import pickle as pickle # python pkl 文件读写

import myCNNs_model

if __name__ == "__mian__":

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




def main(unused_argv):

    train_data = np.array(pickle.load(open('Model/train_data.plk', 'rb')) )
    train_labels = np.array(pickle.load(open('Model/train_labels.plk', 'rb')) )

    eval_data = np.array(pickle.load(open('Model/eval_data.plk', 'rb')) )
    eval_labels = np.array(pickle.load(open('Model/eval_labels.plk', 'rb')) )

    # with tf.Session() as sess:
    #     train_data = tf.convert_to_tensor(train_data_np)
    #     eval_data = tf.convert_to_tensor(eval_data_np)

    # Create the Estimator
    cnn_classifier = tf.estimator.Estimator(
        # model_fn=cnn_model_fn, model_dir="cnn_convnet_model")
        model_fn=cnn_model_fn, model_dir="Model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=50,
        num_epochs=None,
        shuffle=True)
    cnn_classifier.train(
        input_fn=train_input_fn,
        steps=200,     # steps=20000,
        hooks=[logging_hook])

    # 保存模型
    cnn_classifier.export_savedmodel("Model/cnn_model.ckpt", train_input_fn)

    '''
    # 加载模型
    saver = tf.train.Saver()  
    
    with tf.Session() as sess:  
        saver.restore(sess, "./Model/cnn_model.ckpt") # 注意此处路径前添加"./"  
    '''

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False) 
    eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    
    tf.app.run()

