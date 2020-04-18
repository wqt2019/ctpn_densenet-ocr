from __future__ import print_function

import cv2
import glob
import shutil
import sys

import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    with open('./results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("./results", base_name), img)


def draw_yolo_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    with open('./results/' + '{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)

            min_x = int(box[0])
            min_y = int(box[1])
            max_x = int(box[6])
            max_y = int(box[7])

            # cv2.rectangle(img,(min_x,min_y),(max_x,max_y),color)

            center_x = (min_x + max_x) / (2.0 * img.shape[1])
            center_y = (min_y + max_y) / (2.0 * img.shape[0])
            w = (max_x - min_x) / (1.0 * img.shape[1])
            h = (max_y - min_y) / (1.0 * img.shape[0])

            line = '0 ' + ' '.join([str(center_x), str(center_y), str(w), str(h)]) + '\r\n'
            f.write(line)

    cv2.imwrite(os.path.join("./results", base_name), img)



def ctpn(sess, net, image_name):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img_w = img.shape[1]
    img_h = img.shape[0]
    if img_w>=200 and img_h >=200:
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(sess, net, img)

        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        # draw_boxes(img, image_name, boxes, scale)
        draw_yolo_boxes(img, image_name, boxes, scale)
        timer.toc()
        # print(('Detection took {:.3f}s for '
        #        '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

import time

if __name__ == '__main__':
    if os.path.exists("./results/"):
        shutil.rmtree("./results/")
    os.makedirs("./results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'img', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'img', '*.jpg'))

    start=time.time()
    now = time.localtime()
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
    print('start time: ', now_time)
    print('image number: ',len(im_names))

    num = 0
    for im_name in im_names:
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)
        if num% 100 == 0:
            now = time.localtime()
            now_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
            print('time: ', now_time)
            print('image number: ', num)
        num += 1

    print('done...')
    print('time: ',time.time()-start)
    now = time.localtime()
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
    print('end time: ', now_time)
