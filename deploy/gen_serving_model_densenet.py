#-*- coding:utf-8 -*-

from keras import backend as K

import os
import shutil
import tensorflow as tf

import keys
import densenet
from keras.layers import Input
from keras.models import Model
from keras.utils import multi_gpu_model


def decode(pred_text_tensor,axis=2):

    pred_text_index = tf.argmax(pred_text_tensor,axis=axis)
    pred_text_score = tf.reduce_max(pred_text_tensor,axis=axis)

    return pred_text_index,pred_text_score


# densenet的图像预处理已经在ctpn中完成
def preprocess_image(im):

    #等比例将图像高度缩放到32
    im=tf.image.rgb_to_grayscale(im)
    im_shape = tf.shape(im)
    h=im_shape[1]
    w=im_shape[2]
    height=tf.constant(32,tf.int32)
    scale = tf.divide(tf.cast(h,tf.float32),32)
    width = tf.divide(tf.cast(w,tf.float32),scale)
    width =tf.cast(width,tf.int32)

    resize_image = tf.image.resize_images(im, [height,width], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    resize_image = tf.cast(resize_image, tf.float32) / 255 - 0.5

    width = tf.reshape(width, [1])
    height = tf.reshape(height, [1])

    im_info = tf.concat([height, width], 0)
    im_info = tf.concat([im_info, [1]], 0)
    im_info = tf.reshape(im_info, [1, 3])
    im_info = tf.cast(im_info, tf.float32)

    return resize_image,im_info


if __name__ == "__main__":

    # --------
    # raw_image = tf.placeholder(tf.float32, shape=[None,None, None, 1])  #输入图像
    # resize_image,im_info=preprocess_image(raw_image)  #预处理

    version = 1
    path='densenet_savemodel'
    densenet_file = './models/densenet/densenet_single_gpu.h5'

    export_path = './densenet_savemodel/1/'
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    if not os.path.exists(path):
        os.mkdir(path)

    reload(densenet)
    characters = keys.alphabet[:]
    characters = characters[1:] + u'卍'
    nclass = len(characters)

    input = Input(shape=(32, None, 1), name='the_input')
    y_pred= densenet.dense_cnn(input, nclass)
    basemodel = Model(inputs=input, outputs=y_pred)

    modelPath = densenet_file
    # basemodel = multi_gpu_model(basemodel, gpus=2)
    basemodel.load_weights(modelPath)

    #---------
    K.set_learning_phase(0)
    export_path = os.path.join(tf.compat.as_bytes(path),tf.compat.as_bytes(str(version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    print('input info:', basemodel.input_names, '---', basemodel.input)
    print('output info:', basemodel.output_names, '---', basemodel.output)

    input_tensor = basemodel.input
    output_tensor=basemodel.output

    output_index_tensor,output_score_tensor = decode(output_tensor,axis=2) #返回字符索引及其置信度分数，转中文字符在客户端完成

    model_input = tf.saved_model.utils.build_tensor_info(input_tensor)
    model_output = tf.saved_model.utils.build_tensor_info(output_tensor)
    model_output_index = tf.saved_model.utils.build_tensor_info(output_index_tensor)
    model_score_index = tf.saved_model.utils.build_tensor_info(output_score_tensor)


    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': model_input},
            outputs={'prediction': model_output,
                     'prediction_index': model_output_index,
                     'prediction_score': model_score_index,},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))


    with K.get_session() as sess:

        builder.add_meta_graph_and_variables(
            sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict_images':prediction_signature,}
            )

        builder.save()

    print('done...')

