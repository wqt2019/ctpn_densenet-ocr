#-*- coding: utf-8 -*-


import tensorflow as tf
import argparse
import os
import shutil

from lib.rpn_msr.proposal_layer_tf_api import proposal_layer_tf_api


out_save_model = 'ctpn_savemodel'
pb_file = './models/ctpn.pb'
model_version = 1

parser = argparse.ArgumentParser(description='Generate a saved model.')
parser.add_argument('--export_model_dir', type=str, default=out_save_model, help='export model directory')
parser.add_argument('--model_version', type=int, default=model_version, help='model version')
parser.add_argument('--model', type=str, default=pb_file, help='model pb file')

args = parser.parse_args()

if os.path.exists(out_save_model):
    shutil.rmtree(out_save_model)

#------------------

def preprocess_image(image_buffer):

    im = tf.image.decode_jpeg(image_buffer, channels=3)
    im_shape = tf.shape(im)
    src_h, src_w = im_shape[0], im_shape[1]
    im = tf.expand_dims(im, 0)

    #缩放后的图像的宽高限制在600--1200之间
    MIN_SCALE = tf.constant(600,tf.float32)  #600
    MAX_SCALE = tf.constant(1200, tf.float32) #1200
    PIXEL_MEANS = tf.constant([102.9801, 115.9465, 122.7717],tf.float32)
    src_img = tf.cast(im, dtype=tf.float32)
    # src_img -= PIXEL_MEANS

    def resize_im(im, min_scale, max_scale):
        im_shape  = tf.shape(im)
        min_shape = tf.minimum(im_shape[1],im_shape[2])
        max_shape = tf.maximum(im_shape[1],im_shape[2])
        min_shape = tf.cast(min_shape, tf.float32)
        max_shape = tf.cast(max_shape, tf.float32)

        tmp_f = tf.divide(min_scale,min_shape)
        a = tf.multiply(tmp_f, max_shape)

        fn1 = lambda: (tmp_f)
        fn2 = lambda: (tf.divide(max_scale,max_shape))
        f = tf.cond(a > max_scale, true_fn=fn2, false_fn=fn1)


        h = tf.multiply(f, tf.cast(im_shape[1], tf.float32))
        w = tf.multiply(f, tf.cast(im_shape[2], tf.float32))
        w = tf.cast(w, tf.int32)
        h = tf.cast(h, tf.int32)

        resize_image = tf.image.resize_images(im,[h, w],method=tf.image.ResizeMethod.BILINEAR,align_corners=True)
        return resize_image,w,h

    resize_img, w,h = resize_im(src_img, MIN_SCALE, MAX_SCALE)

    w = tf.reshape(w, [1])
    h = tf.reshape(h, [1])
    im_info = tf.concat([h, w],0)
    im_info = tf.concat([im_info, [1]], 0)
    im_info = tf.reshape(im_info, [1, 3])
    im_info = tf.cast(im_info,tf.float32)

    return resize_img,im_info


#-------tf_api--------
def postprocess(cls_prob_tf, box_pred_tf , im_info_tf):
    boxes=proposal_layer_tf_api(cls_prob_tf, box_pred_tf, im_info_tf)
    return boxes



#----- im:输入的图像是缩放到600-1200 ------------
#-------返回裁剪缩放padding后的文本图像，高为32，适配densenet识别网络------------
def crop_resize_image(im,boxes):

    im = tf.image.rgb_to_grayscale(im)

    box_width = boxes[:, 6] - boxes[:, 0]
    box_height = boxes[:, 7] - boxes[:, 1]
    scale = box_height / 32.0
    tmp_widht = box_width / scale
    #裁剪缩放后，将文本框padding到同一大小，组成一个batch
    #文本检测为空时，max_width值不定
    max_width = tf.maximum(tf.cast(tf.reduce_max(tmp_widht), tf.int32),10)

    def resize_pad_image(im):
        # 等比例将图像高度缩放到32
        # im=tf.image.rgb_to_grayscale(im)
        im_shape = tf.shape(im)
        h = im_shape[1]
        w = im_shape[2]
        height = tf.constant(32, tf.int32)
        scale = tf.divide(tf.cast(h, tf.float32), 32)
        width = tf.divide(tf.cast(w, tf.float32), scale)
        width = tf.cast(width, tf.int32)

        resize_image = tf.image.resize_images(im, [height, width], method=tf.image.ResizeMethod.BILINEAR,
                                              align_corners=True)
        resize_image = tf.cast(resize_image, tf.float32) / 255 - 0.5
        #---padding---
        pad_img = tf.image.resize_image_with_crop_or_pad(resize_image, 32, max_width)

        # tmp_width = tf.reshape(max_width, [1])
        # height = tf.reshape(height, [1])
        #
        # crop_re_im_info = tf.concat([height, tmp_width], 0)
        # crop_re_im_info = tf.concat([crop_re_im_info, [1]], 0)
        # crop_re_im_info = tf.reshape(crop_re_im_info, [1, 3])
        # crop_re_im_info = tf.cast(crop_re_im_info, tf.float32)

        return pad_img

    def my_cond(loop_i, tmp_img_input):
        return tf.less(loop_i, loop_len)

    def my_body(loop_i, tmp_img_input):
        cur_box = boxes[loop_i]
        xmin = tf.maximum(tf.cast(cur_box[0], tf.int32),0)
        ymin = tf.maximum(tf.cast(cur_box[1], tf.int32),0)
        xmax = tf.cast(cur_box[6], tf.int32)
        ymax = tf.cast(cur_box[7], tf.int32)
        # score = cur_box[8]

        crop_image = im[:, ymin:ymax, xmin:xmax, :]
        crop_re_img = resize_pad_image(crop_image)
        result = tf.concat([tmp_img_input, tf.reshape(crop_re_img, shape=[1, 32, max_width, 1])], 0)

        return loop_i + 1, result

    loop_len = tf.shape(boxes)[0]
    i = tf.constant(0)
    tmp_img = tf.zeros((1, 32, max_width, 1), tf.float32)
    _, tmp_result_img = tf.while_loop(cond=my_cond, body=my_body,
                                      loop_vars=[i, tmp_img],
                                      shape_invariants=[tf.TensorShape(None), tf.TensorShape([None, 32, None, 1])])

    result_crop_resize_img = tmp_result_img[1:]

    max_width = tf.reshape(max_width,[1])
    fix_h = tf.constant(32, shape=[1])
    crop_resize_im_info =  tf.concat([fix_h, max_width],0)
    crop_resize_im_info = tf.concat([crop_resize_im_info, [1]], 0)
    crop_resize_im_info = tf.reshape(crop_resize_im_info, [1, 3])
    crop_resize_im_info = tf.cast(crop_resize_im_info, tf.float32)

    return result_crop_resize_img,crop_resize_im_info




if __name__ == "__main__":

    with tf.Session() as sess:
        with tf.gfile.GFile(args.model, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())
        tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )

        # #打印节点信息
        # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # for tensor_name in tensor_name_list:
        #     print(tensor_name, '\n')


        export_path_base = args.export_model_dir
        export_path = os.path.join(tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(args.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)


        # raw_image =  tf.placeholder(tf.float32, shape=[None, None, None, 3])  #输入原始图像
        raw_image = tf.placeholder(tf.string, name='tf_image_string')
        jpeg,im_info = preprocess_image(raw_image)  #预处理,缩放

        output_tensor_cls_prob,output_tensor_box_pred = tf.import_graph_def\
                                    (tf.get_default_graph().as_graph_def(),
                                   input_map={'Placeholder:0': jpeg},
                                   return_elements=['Reshape_2:0','rpn_bbox_pred/Reshape_1:0'])

        tensor_info_input = tf.saved_model.utils.build_tensor_info(raw_image)
        tensor_info_output_cls_prob = tf.saved_model.utils.build_tensor_info(output_tensor_cls_prob)
        tensor_info_output_box_pred = tf.saved_model.utils.build_tensor_info(output_tensor_box_pred)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_input},
                outputs={'cls_prob': tensor_info_output_cls_prob,
                         'box_pred': tensor_info_output_box_pred,
                         },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            ))

        im_info_output = tf.saved_model.utils.build_tensor_info(im_info)

        print('\n')
        print('im_info_input tensor shape', im_info.shape)

        # ctpn后处理,合并宽度为16的boxes,得到最终的文本框
        result_boxes=postprocess(output_tensor_cls_prob,output_tensor_box_pred,im_info)


        #-----根据检测的文本框裁剪图像，等比例缩放到高32，padding到同一大小,组batch,适配densenet识别网络----------
        crop_resize_img,crop_resize_im_info = crop_resize_image(jpeg, result_boxes)
        output_crop_resize_img = tf.saved_model.utils.build_tensor_info(crop_resize_img)
        output_crop_resize_img_info = tf.saved_model.utils.build_tensor_info(crop_resize_im_info)
        #----------

        tensor_info_output_boxes = tf.saved_model.utils.build_tensor_info(result_boxes)

        prediction_post_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_input},
                outputs={'detection_boxes': tensor_info_output_boxes,
                         'resize_im_info':im_info_output,
                         'crop_resize_img': output_crop_resize_img,
                         'crop_resize_im_info': output_crop_resize_img_info,
                         },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            ))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':prediction_signature,
                'predict_images_post': prediction_post_signature,
            })


        builder.save(as_text=False)
    print('Done exporting!')

