
# -*- coding:utf-8 -*-
import numpy as np
from .generate_anchors import generate_anchors

from .generate_anchors_tf_api import generate_anchors_tf_api

# from lib.fast_rcnn.config import cfg
# from lib.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
# from lib.fast_rcnn.nms_wrapper import nms
# from .text_proposal_graph_builder import TextProposalGraphBuilder

from lib.text_connector.text_connect_cfg import Config as TextLineCfg


# from lib.text_connector.text_proposal_connector_oriented import TextProposalConnector
from lib.text_connector.text_proposal_connector_oriented_tf_api import TextProposalConnector
# from lib.fast_rcnn.nms_wrapper import py_cpu_nms_tf_api

#------------
import tensorflow as tf




def proposal_layer_tf_api(rpn_cls_prob_reshape, rpn_bbox_pred, im_info):

    # anchor_scales = [16, ]
    # cfg_key=cfg_key.decode('ascii')
    # _anchors = generate_anchors(scales=np.array(anchor_scales))#生成基本的9个anchor
    # _num_anchors = _anchors.shape[0]  # 9个anchor
    _anchors = generate_anchors_tf_api()
    _num_anchors = tf.shape(_anchors)[0]
    # _num_anchors = 10


    im_info = im_info[0]    #原始图像的高宽、缩放尺度



    # pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N#12000,在做nms之前，最多保留的候选box数目
    # post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N#2000，做完nms之后，最多保留的box的数目
    # nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH#nms用参数，阈值是0.7
    # min_size      = cfg[cfg_key].RPN_MIN_SIZE#候选box的最小尺寸，目前是16，高宽均要大于16
    # #TODO 后期需要修改这个最小尺寸，改为8？

    pre_nms_topN=12000 #12000
    post_nms_topN=3000 #1000
    nms_thresh=0.7
    min_size=8
    feat_stride = 16


    #feature-map的高宽
    height = tf.to_int32(tf.floor(im_info[0] / feat_stride))
    width = tf.to_int32(tf.floor(im_info[1] / feat_stride))


    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    # (1, H, W, A)
    tmp_reshape = tf.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])
    tmp_reshape1 = tmp_reshape[:,:,:,:,1]
    scores = tf.reshape(tmp_reshape1,[1, height, width, _num_anchors])
    # return tmp_reshape


    #提取到object的分数，non-object的我们不关心
    #并reshape到1*H*W*9
    bbox_deltas = rpn_bbox_pred     #模型输出的pred是相对值，需要进一步处理成真实图像中的坐标

    # Enumerate all shifts
    # 同anchor-target-layer-tf这个文件一样，生成anchor的shift，进一步得到整张图像上的所有anchor

    _feat_stride = 16

    shift_x = tf.range(0, width * feat_stride, feat_stride)
    shift_y = tf.range(0, height * feat_stride, feat_stride)
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, [-1, 1])
    shift_y = tf.reshape(shift_y, [-1, 1])
    shifts = tf.concat((shift_x, shift_y, shift_x, shift_y), 1)

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors


    anchors = _anchors[tf.newaxis] + tf.transpose(shifts[tf.newaxis], (1, 0, 2))
    anchors = tf.cast(tf.reshape(anchors, (-1, 4)), tf.int32)
    #这里得到的anchor就是整张图像上的所有anchor

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = tf.reshape(bbox_deltas,(-1, 4)) #(HxWxA, 4)

    # Same story for the scores:
    scores = tf.reshape(scores,(-1, 1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv_tf_api(anchors, bbox_deltas)#做逆变换，得到box在图像上的真实坐标

    # 2. clip predicted boxes to image
    proposals = clip_boxes_tf_api(proposals, im_info[:2])#将所有的proposal修建一下，超出图像范围的将会被修剪掉

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes_tf_api(proposals, min_size * im_info[2])#移除那些proposal小于一定尺寸的proposal

    proposals = tf.gather(proposals,keep)#保留剩下的proposal
    scores = tf.gather(scores,keep)
    bbox_deltas=tf.gather(bbox_deltas,keep)


    # # 4. sort all (proposal, score) pairs by score from highest to lowest
    # # 5. take top pre_nms_topN (e.g. 6000)
    # #score按得分的高低进行排序
    # #保留12000个proposal进去做nms

    # if pre_nms_topN > 0:
    #     topN = tf.minimum(tf.shape(scores)[0], pre_nms_topN)
    # else:
    #     topN = tf.shape(scores)[0]

    topN = tf.minimum(tf.shape(scores)[0], pre_nms_topN)
    scores_topn, order = tf.nn.top_k(scores[:, 0], topN, sorted=True)

    proposals = tf.gather(proposals,order)
    scores = tf.gather(scores,order)
    bbox_deltas=tf.gather(bbox_deltas,order)

    # # 6. apply nms (e.g. threshold = 0.7)
    # # 7. take after_nms_topN (e.g. 300)
    # # 8. return the top proposals (-> RoIs top)

    # Non-maximal suppression
    # topN = post_nms_topN if post_nms_topN > 0 else -1

    topN = post_nms_topN
    scores1=tf.reshape(scores,[-1])

    # tf-serving下，多客户端请求时，tf.image.non_max_suppression算法会导致错误
    # 此处nms非必须
    # keep = tf.image.non_max_suppression(proposals, scores1, topN, iou_threshold=nms_thresh)
    # proposals = tf.gather(proposals, keep)
    # scores = tf.gather(scores, keep)
    # bbox_deltas = tf.gather(bbox_deltas, keep)

    # tmp_proposals = tf.cast(proposals, tf.float32)
    # tmp_scores1 = tf.cast(scores1, tf.float32)
    # tmp_scores1 = tf.reshape(tmp_scores1, [-1, 1])
    # dets_tf = tf.concat([tmp_proposals, tmp_scores1], 1)
    # keep = py_cpu_nms_tf_api(dets_tf, nms_thresh)
    # proposals = tf.gather(proposals, keep)
    # scores = tf.gather(scores, keep)
    # bbox_deltas = tf.gather(bbox_deltas, keep)
    #
    # scores_topn, keep = tf.nn.top_k(scores[:, 0], topN, sorted=True)
    # proposals = tf.gather(proposals, keep)
    # scores = tf.gather(scores, keep)
    # bbox_deltas = tf.gather(bbox_deltas, keep)



    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    rois = tf.concat([scores, proposals], 1) #rois

    result_boxes=detect_tf_api(rois,im_info) #得到检测的rois，进行后处理，返回最终的检测结果,tf-api实现
    # 返回的值result_boxes为N * (x1, y1, x2, y2, x3, y3, x4, y4, score)

    return result_boxes



def clip_boxes_tf_api(boxes, im_shape):
    """
    Clip boxes to image boundaries with tensorflow. Note here we assume
    that boxes is always of shape (n, 4).
    """
    clipped_boxes = tf.concat([
        # x1 >= 0
        tf.maximum(tf.minimum(boxes[:, 0:1], im_shape[1] - 1), 0),
        # y1 >= 0
        tf.maximum(tf.minimum(boxes[:, 1:2], im_shape[0] - 1), 0),
        # x2 < im_shape[1]
        tf.maximum(tf.minimum(boxes[:, 2:3], im_shape[1] - 1), 0),
        # y2 < im_shape[0]
        tf.maximum(tf.minimum(boxes[:, 3:4], im_shape[0] - 1), 0)], 1)
    return clipped_boxes


def bbox_transform_inv_tf_api(boxes, deltas):
    """
    TF implementation of bbox_transform_inv. Note here we assume
    that boxes and deltas are always of shape (n, 4).
    """
    # boxes = tf.cast(boxes, deltas.dtype) # TODO maybe remove?
    boxes = tf.cast(boxes, tf.float32)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights

    pred_boxes = tf.transpose(tf.stack([
        # x1
        pred_ctr_x - 0.5 * pred_w,
        # y1
        pred_ctr_y - 0.5 * pred_h,
        # x2
        pred_ctr_x + 0.5 * pred_w,
        # y2
        pred_ctr_y + 0.5 * pred_h,]))

    return pred_boxes

def _filter_boxes_tf_api(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = tf.where((ws >= min_size) & (hs >= min_size))[:,0]
    return keep





def detect_tf_api(rois_tf,im_info):

    scores_tf = rois_tf[:, 0]
    text_proposals_tf = rois_tf[:, 1:5]
    # 删除得分较低的proposal
    keep_inds=tf.where(scores_tf>TextLineCfg.TEXT_PROPOSALS_MIN_SCORE)[:,0]
    text_proposals = tf.gather(text_proposals_tf,keep_inds)
    scores = tf.gather(scores_tf, keep_inds)

    # 按得分排序
    top_nn=tf.shape(scores)[0]
    scores_topn,sorted_indices = tf.nn.top_k(scores,top_nn,sorted=True)
    text_proposals = tf.gather(text_proposals,sorted_indices)
    scores = tf.gather(scores, sorted_indices)
    scores1=tf.reshape(scores,[-1])

    # # 对proposal做nms
    # keep = tf.image.non_max_suppression(text_proposals, scores1, top_nn, iou_threshold=TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)
    # keep=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

    # tf-api实现 nms
    tmp_text_proposals = tf.cast(text_proposals,tf.float32)
    tmp_scores1 = tf.cast(scores1, tf.float32)
    tmp_scores1 = tf.reshape(tmp_scores1,[-1,1])
    dets_tf = tf.concat([tmp_text_proposals, tmp_scores1],1)
    keep = py_cpu_nms_tf_api(dets_tf, TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)



    text_proposals = tf.gather(text_proposals, keep)
    scores = tf.gather(scores, keep)
    scores = tf.reshape(scores, [-1,1])
    # post_blob = tf.concat([scores, text_proposals], 1)


    # # 获取检测结果
    #-------tf-api实现-------------
    textdetector = TextProposalConnector()
    text_recs_tensor = textdetector.get_text_lines_tf_api(text_proposals, scores, im_info)
    keep_inds = filter_finnal_boxes_tf_api(text_recs_tensor)
    result_boxes = tf.gather(text_recs_tensor,keep_inds)
    #返回的值result_boxes为   N*(x1,y1,x2,y2,x3,y3,x4,y4,score)


    return result_boxes



def filter_finnal_boxes_tf_api(boxes):

    def my_cond(loop_i, boxes_tf_input, tmp_tf_input):
        return tf.less(loop_i, loop_len)

    def my_body(loop_i, boxes_tf_input, tmp_tf_input):
        box_tf = boxes_tf_input[loop_i]
        height = (tf.abs(box_tf[5] - box_tf[1]) + tf.abs(box_tf[7] - box_tf[3])) / 2.0 + 1
        width  = (tf.abs(box_tf[2] - box_tf[0]) + tf.abs(box_tf[6] - box_tf[4])) / 2.0 + 1
        score  = box_tf[8]

        height = tf.reshape(height, [-1])
        width = tf.reshape(width, [-1])
        score = tf.reshape(score, [-1])

        boxes_tf_tmp = tf.concat([height, width, score], 0)
        result = tf.concat([tmp_tf_input, tf.reshape(boxes_tf_tmp, shape=[1, 3])], 0)

        return loop_i + 1,boxes_tf_input,  result

    loop_len = tf.shape(boxes)[0]
    i = tf.constant(0)
    tmp_tf = tf.zeros((1, 3), tf.float32)
    _, _, tmp_boxes_tf = tf.while_loop(cond=my_cond, body=my_body, loop_vars=[i, boxes, tmp_tf],
                        shape_invariants=[tf.TensorShape(None), tf.TensorShape([None, 9]),tf.TensorShape([None, 3])])
    # parallel_iterations=1024, back_prop=False
    result_boxes = tmp_boxes_tf[1:]

    #过滤boxes
    keep = tf.where((tf.divide(result_boxes[:, 1],result_boxes[:, 0])>TextLineCfg.MIN_RATIO)
                     & (result_boxes[:, 2]>TextLineCfg.LINE_MIN_SCORE) &
                    (result_boxes[:, 1]>(TextLineCfg.TEXT_PROPOSALS_WIDTH*TextLineCfg.MIN_NUM_PROPOSALS)))[:,0]

    return keep


# tf-api nms

def py_cpu_nms_tf_api(dets_tf, thresh):
    dets_tf = tf.cast(dets_tf,tf.float32)
    x1 = dets_tf[:, 0]
    y1 = dets_tf[:, 1]
    x2 = dets_tf[:, 2]
    y2 = dets_tf[:, 3]
    scores = dets_tf[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    top_nn = tf.shape(scores)[0]
    scores_topn, order = tf.nn.top_k(scores, top_nn, sorted=True)

    def my_cond(loop_i, order_input,result):
        cur_order_len=tf.shape(order_input)[0]
        flag = tf.cond(tf.equal(cur_order_len, 0), lambda: False, lambda: True)
        return flag

    def my_body(loop_i, order_input,tmp_tf):
        i = order_input[0]
        i = tf.cast(i,tf.int32)
        xx1 = tf.maximum(x1[i], tf.gather(x1, order_input[1:]))
        yy1 = tf.maximum(y1[i], tf.gather(y1, order_input[1:]))
        xx2 = tf.minimum(x2[i], tf.gather(x2, order_input[1:]))
        yy2 = tf.minimum(y2[i], tf.gather(y2, order_input[1:]))
        w = tf.maximum(0.0, xx2 - xx1 + 1)
        h = tf.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + tf.gather(areas, order_input[1:]) - inter)
        inds = tf.where(ovr <= thresh)
        inds = tf.reshape(inds, [-1])
        order_input = tf.gather(order_input, inds + 1)

        result = tf.concat([tmp_tf, tf.reshape(i,[1])], 0)

        return loop_i + 1, order_input,result

    ii = tf.constant(0)
    tmp = tf.constant(0, shape=[1])
    _,_, tmp_result = tf.while_loop(cond=my_cond, body=my_body,
                                      loop_vars=[ii,order, tmp],
                            shape_invariants=[tf.TensorShape(None), tf.TensorShape(None),tf.TensorShape(None)])

    keep = tmp_result[1:]
    keep = tf.reshape(keep,[-1])

    return keep

