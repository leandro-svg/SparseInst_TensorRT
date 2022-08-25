# SparseInst_TensorRT
 This repository implement the real-time Instance Segmentation Algorithm named SparseInst with TensoRT and ONNX.
 
**Some remarks** 
  - The initial repository on which I build mine is from hustvl/SparseInst repository (https://github.com/hustvl/SparseInst.git), for additional information about the installation of SparseInst, refer to the original repository. 
  - This project is built upon the excellent framework detectron2, and you should install detectron2 first, please check official installation guide for more details. (https://github.com/facebookresearch/detectron2.git)
  - For command other than TensoRT and ONNX inference, please refer to the initial repository (e.g eval.py). 
  - If you face any problem during the parsing time, don't hesitate to drop an issue or a star if there aren't any at all :stuck_out_tongue_winking_eye:	
  
 
 **Prerequisites**
  - Install Pytorch (1.10.0) and TorchVision (1.11)
  - Install CUDA (10.2) and cuDNN (8.0.0)
  - Install TensorRT (8.0.1.6), if you are using an nvidia edge device, TensorRT should already be installed
  - Install ONNX and ONNXruntime
  - Install all the other packages needed to run the original SparseInst algorithm (Should be done if you have installed Dectectron2)
  - Please download the weights pytorch file from hustvl/SparseInst :  'weights/sparse_inst_r50_giam_aug_2b7d68.pth'.
 
 
 Be aware that in order to parse the model to ONNX and TensorRT, the files sparseinst.py, encoder.py and decoder.py has been modified/slightly modified, don't forget to check the modifications if you come from the initial repository.
 For now, the code works with all type of input sizes. Further implementations will be added such as multiple images and videos and different input sizes. 
 
 **Result for TensorRT and ONNX inference script:**
 
 The inference speed for Pytorch, ONNX and TensorRT has been compared and a shown in the table below. SparseInst running with TensoRT achieved more a less 3 times faster inference speed of SparseInst running with Pytorch.
 
 Note: All the computations has been done on a Nvidia Jetson TX2 Jetpack 4.6.
 
 ```
 Pytorch use time 58.41329765319824 for loop 100, FPS= 1.7119389594078978
 TRT use time 19.569902181625366for loop 100, FPS=5.109887574905321
 ONNX use time 3314.860335588455 for loop 100, FPS= 0.03016718349379506 
 ```
 
 <img
  src="results/result_tensorrt.png"
  alt="Alt text"
  title="Result for TensorRT demo"
  style="display: inline-block; margin: 1 auto; max-width: 150px">
 
 
 
 
 

 **Build the ONNX model  :**
 
 To build the model from Pytorch to ONNX, you need to run the following command. You can set the arguments to default. Please check if the config path and the model weights path are correctly set up.
 ```
 <sudo python3 convert_onnx.py --config-file config-gile --output output_directory_onnxmodel --image dummy_input --opts MODEL.WEIGHTS weights_directory>
 ```
 
  **Build the TensorRT model  :**
  
  To build the model from ONNX to TensorRT, you need to run the following command. You can set the arguments to default. If you have any problem while parsing the model to TensorRT, don't hesitate to ask.
 ```
 <sudo python3 convert_tensortt.py --onnx_model onnx-model-directory --output output_directory_TensoRTModel
 ```
 
  **Testing SparseInt with Pytorch, TensorRT and ONNX :**
  
  To test the inference speed (FPS) of the Pytorch, TensorRT and ONNX models, run the following command. The segmentation results for all three will be stored in the resutls directory. You can set all the arguments on default.
 ```
 <sudo python3 eval_tensorrt_onnx.py --config-file config_file_directory --onnx_engine onnx-model-directory --tensorRT_engine tensorRT-model-directory --input input_image_directory 
 ```
 
 
 
 



 
