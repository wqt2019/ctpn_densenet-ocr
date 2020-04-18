# ocr: ctpn + densenet  
  
re-implement the pre-process and post-process of CTPN/densenet by pure tensorflow-api(no numpy function,python27)[proposal_layer_tf_api.py](./lib/rpn_msr/proposal_layer_tf_api.py), you can deploy the model(ctpn/densenet/ctpn+densenet) by tensorflow serving.  
  
the Server only include one command:  
  
#ctpn:  
sudo docker run -p 9002:8501 -p 9001:8500 --mount type=bind,source=/xxx/ctpn_densenet/deploy/ctpn_savemodel/,target=/models/ctpn -e MODEL_NAME=ctpn -t tensorflow/serving:1.11.0  
  
#densenet:  
sudo docker run -p 9002:8501 -p 9001:8500 --mount type=bind,source=/xxx/ctpn_densenet/deploy/densenet_savemodel/,target=/models/densenet -e MODEL_NAME=densenet -t tensorflow/serving:1.11.0  
  
#ctpn+densenet:  
sudo docker run -p 9002:8501 -p 9001:8500 --mount type=bind,source=/xxx/ctpn_densenet/deploy/ctpn_densenet/,target=/models/ocr -e MODEL_NAME=ocr -t tensorflow/serving:1.11.0  
  
  
#train  
1.https://github.com/eragonruan/text-detection-ctpn  
2.https://github.com/YCG09/chinese_ocr  
  
  
#deploy  
cd deploy  
1.python [generate_ctpn_pb.py](./deploy/generate_ctpn_pb.py) . convert ckpt to pb  
2.python [gen_serving_model_ctpn.py](./deploy/gen_serving_model_ctpn.py)  . convert pb to savemodel  
3.python [gen_serving_model_densenet.py](./deploy/gen_serving_model_densenet.py)  . convert keras-model to savemodel  
4.python /xxx/tensorflow/python/tools/saved_model_cli.py show --dir /xxx/ctpn_densenet/deploy/ctpn_savemodel/1/ --all  
  python /xxx/tensorflow/python/tools/saved_model_cli.py show --dir /xxx/ctpn_densenet/deploy/densenet_savemodel/1/ --all  
 check the input and output of each savemodel, and then edit merge_ctpn_densenet_savemodel.py   
5.python [merge_ctpn_densenet_savemodel.py](./deploy/merge_ctpn_densenet_savemodel.py)  . merge the ctpn_savemodel and densenet_savemodel into one model  
6.Server : sudo docker run -p 9002:8501 -p 9001:8500 --mount type=bind,source=/xxx/ctpn_densenet/deploy/ctpn_densenet/,target=/models/ocr -e MODEL_NAME=ocr -t tensorflow/serving:1.11.0  
7.python [ocr_client.py](./deploy/ocr_client.py)   
  
also, you can debug the post-process(tensorflow-api implementation) of CTPN with [demo_ctpn_pb.py](./deploy/demo_ctpn_pb.py) .  
  
  
  
#reference:  
https://github.com/eragonruan/text-detection-ctpn  
https://github.com/YCG09/chinese_ocr  
  
  
