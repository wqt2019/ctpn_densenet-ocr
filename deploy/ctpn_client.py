#-*- coding: utf-8 -*-

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf
import numpy as np
import cv2,os

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import scipy.misc

MAX_MESSAGE_LENGTH = -1
imagedata = []

# sudo docker run -p 9002:8501 -p 9001:8500 --mount type=bind,
# source=/xxx/model/,target=/models/ctpn -e MODEL_NAME=ctpn -t tensorflow/serving:1.11.0



def predict_image(save_path,file_name):

    src_img = cv2.imread(file_name)

    with open(file_name, 'rb') as f:
        img_str = f.read()

        hostport='localhost:9001'
        channel = grpc.insecure_channel(hostport, options=
                                [('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                 ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'ctpn'
        request.model_spec.signature_name = 'predict_images_post'
        request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(img_str))
        result_future = stub.Predict.future(request)

        boxes = result_future.result().outputs['detection_boxes']  #返回文本框结果--(x1,y1,x2,y2,x3,y3,x4,y4,score)
        boxes_list = (tf.contrib.util.make_ndarray(boxes).tolist())
        im_info = result_future.result().outputs['resize_im_info']  #缩放后的图像大小,服务端实现图像缩放
        im_info_list = (tf.contrib.util.make_ndarray(im_info).tolist())
        im_w = int(im_info_list[0][1])
        im_h = int(im_info_list[0][0])

        show_img = cv2.resize(src_img, (im_w, im_h), interpolation=cv2.INTER_CUBIC)

        #---- show-image ---------
        thresh=0.7
        length=len(boxes_list)
        j=0
        for i in range(length):
            cur_roi=boxes_list[i]
            score = cur_roi[-1]
            if(score > thresh):
                if score >= 0.9:
                    color = (0, 255, 0)
                elif score >= 0.8:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 0)

                x1 = int(cur_roi[0])
                y1 = int(cur_roi[1])
                x2 = int(cur_roi[2])
                y2 = int(cur_roi[3])
                x3 = int(cur_roi[4])
                y3 = int(cur_roi[5])
                x4 = int(cur_roi[6])
                y4 = int(cur_roi[7])

                cv2.line(show_img, (x1, y1), (x2, y2), color, 2)
                cv2.line(show_img, (x1, y1), (x3, y3), color, 2)
                cv2.line(show_img, (x2, y2), (x4, y4), color, 2)
                cv2.line(show_img, (x3, y3), (x4, y4), color, 2)

                show_text=str(round(score,3))
                cv2.putText(show_img, show_text, (x1, y1+2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                j+=1

        cv2.imwrite(save_path + 'result.jpg', show_img)

        prediction = 'ok'

        #---------------
        crop_resize_img = result_future.result().outputs['crop_resize_img']  # 返回文本框结果--(x1,y1,x2,y2,x3,y3,x4,y4,score)
        crop_resize_img_list = (tf.contrib.util.make_ndarray(crop_resize_img).tolist())
        crop_resize_img_np = np.array(crop_resize_img_list)

        crop_resize_im_info = result_future.result().outputs['crop_resize_im_info']  # 缩放后的图像大小,服务端实现图像缩放
        crop_resize_im_info_list = (tf.contrib.util.make_ndarray(crop_resize_im_info).tolist())
        crop_resize_im_w = int(crop_resize_im_info_list[0][1])
        crop_resize_im_h = int(crop_resize_im_info_list[0][0])

        print('w,h:',crop_resize_im_w,crop_resize_im_h)
        length = len(crop_resize_img_np)
        for i in range(length):
            cur_img = crop_resize_img_np[i]
            cur_img = cur_img.reshape([crop_resize_im_h,crop_resize_im_w])
            scipy.misc.imsave(save_path+str(i)+'.jpg', cur_img)

    return prediction


if __name__ == '__main__':
    image_file = './img/1.jpg'
    save_path = 'results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    result = predict_image(save_path ,image_file)
    print('pre:', result)
