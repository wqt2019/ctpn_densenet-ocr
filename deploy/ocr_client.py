#-*- coding: utf-8 -*-


from __future__ import print_function

import grpc
import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

MAX_MESSAGE_LENGTH=-1


import keys
characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

char_score_thresh = 0.3


def decode(pred_text_index,pred_text_score):
    char_list = []
    text_str = ''
    for ii in range(len(pred_text_index)):
        pred_text = pred_text_index[ii]
        pred_score = pred_text_score[ii]
        tmp_list = []
        for i in range(len(pred_text)):
            if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))
                        or (i > 1 and pred_text[i] == pred_text[i - 2])) and pred_score[i] >= char_score_thresh:
                tmp=characters[pred_text[i]]
                tmp_list.append(tmp)

        #去掉因为padding而识别错误的字符，如：| [ ]等
        if len(tmp_list)>=2:
            if tmp_list[0]=='|' or tmp_list[0]=='[' or tmp_list[0]==']':
                del(tmp_list[0])
            if tmp_list[-1]=='|' or tmp_list[-1]=='[' or tmp_list[-1]==']':
                del(tmp_list[-1])

        tmp_text = u''.join(tmp_list)
        char_list.append(tmp_text)
        text_str += tmp_text

    return char_list,text_str



def predict_image(img_file):
    # 缩放在服务端完成 , 输入 [tf_string]

    with open(img_file, 'rb') as f:
        img_str = f.read()

        #----- ctpn文本检测 + densenet字符识别 ----
        hostport = 'localhost:9001'
        # hostport='xx.xx.xx.xx:9001'

        channel = grpc.insecure_channel(hostport,
                    options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])

        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'ocr'
        request.model_spec.signature_name = 'predict'

        # tensor_proto = tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,string_val=[img_str])
        # request.inputs['images'].CopyFrom(tensor_proto)

        request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(img_str))
        result_future = stub.Predict.future(request)

        result_index = result_future.result().outputs['prediction_text_index']  # 返回文本识别索引结果, cn_5990.txt中的索引
        result_index_list = (tf.contrib.util.make_ndarray(result_index).tolist())
        result_index_np = np.array(result_index_list)
        result_score = result_future.result().outputs['prediction_text_score']  # 返回文本识别索引score
        result_score_list = (tf.contrib.util.make_ndarray(result_score).tolist())
        result_score_np = np.array(result_score_list)

        ocr_list, ocr_text = decode(result_index_np, result_score_np)  # 索引转字符

    print('ocr_text:', ocr_text)

    return ocr_text


if __name__ == '__main__':

    image_file = './img/1.jpg'

    start = time.time()
    result = predict_image(image_file)
    stop = time.time()
    print('done...,time:', stop - start)

