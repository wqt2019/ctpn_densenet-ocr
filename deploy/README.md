#ctpn:  
  
sudo docker run -p 9002:8501 -p 9001:8500 --mount type=bind,source=/xxx/ctpn_densenet/deploy/ctpn_savemodel/,target=/models/ctpn -e MODEL_NAME=ctpn -t tensorflow/serving:1.11.0
  
#densenet:  
sudo docker run -p 9002:8501 -p 9001:8500 --mount type=bind,source=/xxx/ctpn_densenet/deploy/densenet_savemodel/,target=/models/densenet -e MODEL_NAME=densenet -t tensorflow/serving:1.11.0
  
#ctpn+densenet:  
sudo docker run -p 9002:8501 -p 9001:8500 --mount type=bind,source=/xxx/ctpn_densenet/deploy/ctpn_densenet/,target=/models/ocr -e MODEL_NAME=ocr -t tensorflow/serving:1.11.0
  
#######################################  
  
python /xxx/tensorflow/python/tools/saved_model_cli.py show --dir /xxx/ctpn_densenet/deploy/ctpn_savemodel/1/ --all
  
python /xxx/tensorflow/python/tools/saved_model_cli.py show --dir /xxx/ctpn_densenet/deploy/densenet_savemodel/1/ --all
  
  
#----ctpn-------------------------  
  
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
  
signature_def['predict_images']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['images'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: tf_image_string:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['box_pred'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1, -1, 80)
        name: import/rpn_bbox_pred/Reshape_1:0
    outputs['cls_prob'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1, -1, 40)
        name: import/Reshape_2:0
  Method name is: tensorflow/serving/predict

signature_def['predict_images_post']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['images'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: tf_image_string:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['crop_resize_im_info'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 3)
        name: Cast_13:0
    outputs['crop_resize_img'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 32, -1, 1)
        name: strided_slice_91:0
    outputs['detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 9)
        name: GatherV2_13:0
    outputs['resize_im_info'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 3)
        name: Cast_7:0
  Method name is: tensorflow/serving/predict

#-----densenet----------------------  
  
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
  
signature_def['predict_images']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['images'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 32, -1, 1)
        name: the_input_2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['prediction'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1, 5990)
        name: out_2/truediv:0
    outputs['prediction_index'] tensor_info:
        dtype: DT_INT64
        shape: (-1, -1)
        name: ArgMax:0
    outputs['prediction_score'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1)
        name: Max:0
  Method name is: tensorflow/serving/predict
  
  
  

