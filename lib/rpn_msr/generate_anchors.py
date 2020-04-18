import numpy as np

def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    # heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    heights = [11, 17, 23, 29, 34, 42, 51, 61, 71, 82, 95, 110, 128, 149, 173, 202, 235, 272, 316, 370]
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes)



#
# import tensorflow as tf
#
# def generate_anchors_tf_api():
#
#     heights = tf.constant([11, 16, 23, 33, 48, 68, 97, 139, 198, 283])
#     widths = tf.constant(16)
#     tmp_size = tf.zeros((1,2), tf.int32)
#
#     def my_cond(loop_i, tmp_size_input):
#         return tf.less(loop_i, loop_len)
#
#     def my_body(loop_i, tmp_size_input):
#         h = tf.reshape(heights[loop_i], shape=[1])
#         w = tf.reshape(widths, shape=[1])
#         tmp = tf.concat([h,w],0)
#         tmp = tf.reshape(tmp, shape=[-1, 2])
#         result = tf.concat([tmp_size_input,tmp],0)
#         result = tf.reshape(result,shape=[-1,2])
#
#         return loop_i + 1, result
#
#     loop_len = tf.shape(heights)[0]
#     i = tf.constant(0)
#     _, tmp_result_size = tf.while_loop(cond=my_cond, body=my_body,
#                                      loop_vars=[i, tmp_size],
#                                      shape_invariants=[tf.TensorShape(None), tf.TensorShape([None,2])])
#
#     sizes = tmp_result_size[1:]
#     final_anchors = generate_basic_anchors_tf_api(sizes, base_size=16)
#
#     return final_anchors
#
# def generate_basic_anchors_tf_api(sizes, base_size=16):
#     base_anchor = tf.constant([0, 0, base_size - 1, base_size - 1], tf.int32)
#     tmp_size = tf.zeros((1, 4), tf.int32)
#
#     def my_cond(loop_i, tmp_size_input):
#         return tf.less(loop_i, loop_len)
#
#     def my_body(loop_i, tmp_size_input):
#
#         tmp_wh = sizes[loop_i]
#         tmp_h = tmp_wh[0]
#         tmp_w = tmp_wh[1]
#
#         x = tf.cast(base_anchor[0] + base_anchor[2], tf.float32)
#         y = tf.cast(base_anchor[1] + base_anchor[3], tf.float32)
#         x_ctr = x/2
#         y_ctr = y/2
#
#         half_w = tf.cast(tmp_w / 2, tf.int32)
#         half_h = tf.cast(tmp_h / 2, tf.int32)
#         half_w = tf.cast(half_w , tf.float32)
#         half_h = tf.cast(half_h , tf.float32)
#
#         xmin = tf.cast(x_ctr - half_w ,tf.int32) # xmin
#         xmax = tf.cast(x_ctr + half_w  ,tf.int32) # xmax
#         ymin = tf.cast(y_ctr - half_h ,tf.int32) # ymin
#         ymax = tf.cast(y_ctr + half_h ,tf.int32) # ymax
#
#         tmp_anchor = tf.transpose(tf.stack([xmin,ymin,xmax,ymax]))
#         tmp_anchor = tf.reshape(tmp_anchor, shape=[-1, 4])
#
#         result = tf.concat([tmp_size_input, tmp_anchor], 0)
#         result = tf.reshape(result, shape=[-1, 4])
#
#         return loop_i + 1, result
#
#     loop_len = tf.shape(sizes)[0]
#     i = tf.constant(0)
#     _, tmp_result_anchor = tf.while_loop(cond=my_cond, body=my_body,
#                                        loop_vars=[i, tmp_size],
#                                        shape_invariants=[tf.TensorShape(None), tf.TensorShape([None, 4])])
#
#     anchors = tmp_result_anchor[1:]
#     return anchors





if __name__ == '__main__':
    b= generate_anchors_tf_api()


    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
