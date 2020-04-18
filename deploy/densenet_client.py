#-*- coding: utf-8 -*-

# mnist_client.py --num_tests=100 --server=localhost:9000


from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
MAX_MESSAGE_LENGTH = -1


import keys
characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)


char_score_thresh = 0.3
def decode(pred):
    char_list = []
    index_list = []
    tmp_pred_text = pred.argmax(axis=2)
    text_str = ''
    for ii in range(len(tmp_pred_text)):
        pred_text = tmp_pred_text[ii]
        tmp_list = []
        for i in range(len(pred_text)):
            max_index = pred_text[i]
            cur_char_score = pred[ii][i][max_index]
            if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1]))
                        or (i > 1 and pred_text[i] == pred_text[i - 2])) and cur_char_score > char_score_thresh :
                tmp=characters[pred_text[i]]
                tmp_list.append(tmp)
                index_list.append(pred_text[i])
                # char_list.append(characters[pred_text[i]])

        if len(tmp_list)>=2:
            if tmp_list[0]=='|' or tmp_list[0]=='[' or tmp_list[0]==']':
                del(tmp_list[0])
            if tmp_list[-1]=='|' or tmp_list[-1]=='[' or tmp_list[-1]==']':
                del(tmp_list[-1])

        tmp_text = u''.join(tmp_list)
        char_list.append(tmp_text)
        text_str += tmp_text

    return char_list,text_str,index_list


def predict_image(file_name):

    img = np.array(Image.open(file_name))
    img = Image.fromarray(img).convert('L')
    width, height = img.size[0], img.size[1]
    f = height/32.0
    re_w = int(width/f)
    img=img.resize((re_w,32),Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)/255-0.5
    arr_img = img.reshape([-1, 32, re_w, 1])

    #------------------------
    hostport='localhost:9001'
    channel = grpc.insecure_channel(hostport, options=
                                    [('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                     ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'densenet'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(arr_img))
    result_future = stub.Predict.future(request)

    result = result_future.result().outputs['prediction']
    result_list = (tf.contrib.util.make_ndarray(result).tolist())
    result_np=np.array(result_list)

    result_index = result_future.result().outputs['prediction_index']
    result_index_list = (tf.contrib.util.make_ndarray(result_index).tolist())
    result_index_np = np.array(result_index_list)

    text_list, text_str,tmp_index = decode(result_np)

    return text_str


if __name__ == '__main__':
    image_file = './img/2.jpg'
    result = predict_image(image_file)
    print('prediction:', result)



