#-*-coding:utf-8-*-

import tensorflow as tf
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import os, shutil

#先打印savemodel的节点信息，确定输入输出
# python /xxx/tensorflow/tensorflow/python/tools/saved_model_cli.py show --dir /xxx/model/1/ --all

ctpn_path = './ctpn_savemodel/1'
desenet_path = './densenet_savemodel/1/'
ocr_model_path='./ctpn_densenet/1'

if os.path.exists(ocr_model_path):
    shutil.rmtree(ocr_model_path)


if __name__ == "__main__":

    with tf.Graph().as_default() as g1:
        with tf.Session(graph=g1) as sess1:
            input_graph_def1 = saved_model_utils.get_meta_graph_def(ctpn_path, tf.saved_model.tag_constants.SERVING).graph_def
            tf.saved_model.loader.load(sess1, ["serve"], ctpn_path)
            g1def = convert_variables_to_constants\
                (sess1, input_graph_def1, output_node_names=['strided_slice_91'],
                variable_names_whitelist=None, variable_names_blacklist=None)

    with tf.Graph().as_default() as g2:
        with tf.Session(graph=g2) as sess2:
            input_graph_def2 = saved_model_utils.get_meta_graph_def(desenet_path, tf.saved_model.tag_constants.SERVING).graph_def
            tf.saved_model.loader.load(sess2, ["serve"], desenet_path)
            g2def = convert_variables_to_constants\
                (sess2, input_graph_def2, output_node_names=['ArgMax','Max'],
                variable_names_whitelist=None, variable_names_blacklist=None)



    with tf.Graph().as_default() as g_combined:
        with tf.Session(graph=g_combined) as sess:
            # raw_image = tf.placeholder(tf.float32, shape=[None, None, None, 3],name="images")
            raw_image = tf.placeholder(tf.string, name="images")
            crop_resize_img, = tf.import_graph_def(g1def,
                       input_map={"tf_image_string:0": raw_image}, return_elements=["strided_slice_91:0"])
            text_index, text_score = tf.import_graph_def(g2def, input_map=
                            {"the_input_2:0": crop_resize_img}, return_elements=["ArgMax:0","Max:0"])

            builder = tf.saved_model.builder.SavedModelBuilder(ocr_model_path)
            tensor_info_raw_image = tf.saved_model.utils.build_tensor_info(raw_image)
            tensor_info_text_index = tf.saved_model.utils.build_tensor_info(text_index)
            tensor_info_text_score = tf.saved_model.utils.build_tensor_info(text_score)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_raw_image},
                    outputs={'prediction_text_index': tensor_info_text_index,
                             'prediction_text_score': tensor_info_text_score, },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict': prediction_signature,
                })

            builder.save(as_text=False)

    print('done...')


