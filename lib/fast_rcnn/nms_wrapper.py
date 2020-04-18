import numpy as np
from .config import cfg
pure_python_nms = False
try:
    from lib.utils.gpu_nms import gpu_nms
    from ..utils.cython_nms import nms as cython_nms
except ImportError:
    pure_python_nms = True


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    if pure_python_nms:
        # print("Fall back to pure python nms")
        return py_cpu_nms(dets, thresh)
    if cfg.USE_GPU_NMS:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cython_nms(dets, thresh)


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


#
# # tf-api nms
# import tensorflow as tf
#
# def py_cpu_nms_tf_api(dets_tf, thresh):
#     dets_tf = tf.cast(dets_tf,tf.float32)
#     x1 = dets_tf[:, 0]
#     y1 = dets_tf[:, 1]
#     x2 = dets_tf[:, 2]
#     y2 = dets_tf[:, 3]
#     scores = dets_tf[:, 4]
#
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     top_nn = tf.shape(scores)[0]
#     scores_topn, order = tf.nn.top_k(scores, top_nn, sorted=True)
#
#     def my_cond(loop_i, order_input,result):
#         cur_order_len=tf.shape(order_input)[0]
#         flag = tf.cond(tf.equal(cur_order_len, 0), lambda: False, lambda: True)
#         return flag
#
#     def my_body(loop_i, order_input,tmp_tf):
#         i = order_input[0]
#         i = tf.cast(i,tf.int32)
#         xx1 = tf.maximum(x1[i], tf.gather(x1, order_input[1:]))
#         yy1 = tf.maximum(y1[i], tf.gather(y1, order_input[1:]))
#         xx2 = tf.minimum(x2[i], tf.gather(x2, order_input[1:]))
#         yy2 = tf.minimum(y2[i], tf.gather(y2, order_input[1:]))
#         w = tf.maximum(0.0, xx2 - xx1 + 1)
#         h = tf.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         ovr = inter / (areas[i] + tf.gather(areas, order_input[1:]) - inter)
#         inds = tf.where(ovr <= thresh)
#         inds = tf.reshape(inds, [-1])
#         order_input = tf.gather(order_input, inds + 1)
#
#         result = tf.concat([tmp_tf, tf.reshape(i,[1])], 0)
#
#         return loop_i + 1, order_input,result
#
#     ii = tf.constant(0)
#     tmp = tf.constant(0, shape=[1])
#     _,_, tmp_result = tf.while_loop(cond=my_cond, body=my_body,
#                                       loop_vars=[ii,order, tmp],
#                             shape_invariants=[tf.TensorShape(None), tf.TensorShape(None),tf.TensorShape(None)])
#
#     keep = tmp_result[1:]
#     keep = tf.reshape(keep,[-1])
#
#     return keep

