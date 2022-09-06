# SparseInst_TensorRT
 **This repository implement the real-time Instance Segmentation Algorithm named SparseInst with TensoRT and ONNX.**
 
## Some remarks 
  - The initial repository on which I build mine is from hustvl/SparseInst repository (https://github.com/hustvl/SparseInst.git), for additional information about the installation of SparseInst, refer to the original repository. 
  - This project is built upon the excellent framework detectron2, and you should install detectron2 first, please check official installation guide for more details. (https://github.com/facebookresearch/detectron2.git)
  - For command other than TensoRT and ONNX inference, please refer to the initial repository (e.g test_net.py). 
  - If you face any problem during the parsing time, don't hesitate to drop an issue or a star if there aren't any at all :stuck_out_tongue_winking_eye:	
  
 
 ## Prerequisites
  - Install Pytorch (1.10.0) and TorchVision (0.11.1)
  ```
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  
  If other versions of torch are needed, select yours by putting torch==1.11.0+cu102 for example.
  ```
  - Install CUDA (10.2) and cuDNN (8.0.0) : https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
  
    - For WSL-Ubuntu :
  ```
  sudo wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
  sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  sudo wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_insta
      llers/cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
  sudo cp /var/cuda-repo-wsl-ubuntu-11-7-local/cuda-96193861-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cuda
   ```
  
  - Install TensorRT (8.0.1.6), if you are using an nvidia edge device, TensorRT should already be installed
  ```
  python3 -m pip install --upgrade setuptools pip
  python3 -m pip install nvidia-pyindex
  python3 -m pip install --upgrade nvidia-tensorrt
  
  Verify installation by writing  : assert tensorrt.Builder(tensorrt.Logger())
  ```
  - Install ONNX and ONNXruntime
  ```
  pip install onnxruntime-gpu
  pip install onnxruntime
  pip install numpy protobuf==4.21.5  
  pip install onnx
  ```
  - Install all the other packages needed to run the original SparseInst algorithm (Should be done if you have installed Dectectron2)
  - Please download the weights pytorch file from hustvl/SparseInst :  'weights/sparse_inst_r50_giam_aug_2b7d68.pth'.
 
 
 Be aware that in order to parse the model to ONNX and TensorRT, the files sparseinst.py, encoder.py and decoder.py has been modified/slightly modified, don't forget to check the modifications if you come from the initial repository.
 For now, the code works with all type of input sizes. Further implementations will be added such as multiple images and videos and different input sizes. 
 
 ## Model and Result for TensorRT and ONNX inference script:
 




 
 The inference speed for Pytorch, ONNX and TensorRT has been compared and shown in the table below. SparseInst running with TensoRT achieved more a less 3 times faster inference speed of SparseInst than running with Pytorch.
 
 Note: All the computations has been done on a Nvidia Jetson TX2 Jetpack 4.6.
 
 | Model | Input Size |  Inference Speed| Weights
| :---         |     :---:      |        :---: |         ---: |
| Pytorch   | 640   | 1.71  FPS  | [model](https://drive.google.com/file/d/130gyxYT6r9j5Nwp5nCo_wthYPuTwa9c4/view?usp=sharing)|
| TensorRT     | 320    |  20.32 FPS     |[model](https://drive.google.com/file/d/17-eBWVrpnwv0ueeDsEmAqSKlNh3If4AI/view?usp=sharing)|
| TensorRT     | 640    |  6.00 FPS     |[model](https://drive.google.com/file/d/1Kh97LZNzsuBJTeDVXwRKx8CiX7CeMI3v/view?usp=sharing)|
| ONNX     | 320    | 0.22 FPS     |[model](https://drive.google.com/file/d/1H6YH3YUPaA4vO3IyIGaZNAkGBsU9xHCH/view?usp=sharing)|
| ONNX     | 640     |0.03 FPS     |[model](https://drive.google.com/file/d/1GEoQssyJ9MZRnEISiatF_tREpdGAnSjk/view?usp=sharing)|
 

 
 <img
  src="results/result_tensorrt.png"
  alt="Alt text"
  title="Result for TensorRT demo"
  style="display: inline-block; margin: 1 auto; max-width: 150px">
 
 
 
 
 

 ## Build the ONNX model  :
 
 To build the model from Pytorch to ONNX, you need to run the following command. You can set the arguments to default. Please check if the config path and the model weights path are correctly set up.
 ```
 <sudo python3 convert_onnx.py --config-file config-gile --output output_directory_onnxmodel --image dummy_input --opts MODEL.WEIGHTS weights_directory>
 ```
 
  ## Build the TensorRT model  :
  
  To build the model from ONNX to TensorRT, you need to run the following command. You can set the arguments to default. If you have any problem while parsing the model to TensorRT, don't hesitate to ask.
 ```
 <sudo python3 convert_tensortt.py --onnx_model onnx-model-directory --output output_directory_TensoRTModel
 ```
 
  ## Testing SparseInt with Pytorch, TensorRT and ONNX :
  
  To test the inference speed (FPS) of the Pytorch, TensorRT and ONNX models, run the following command. The segmentation results for all three will be stored in the resutls directory. You can set all the arguments on default.
 ```
 <sudo python3 eval_tensorrt_onnx.py --config-file config_file_directory --onnx_engine onnx-model-directory --tensorRT_engine tensorRT-model-directory --input input_image_directory 
 ```
 
 
 
 



 
