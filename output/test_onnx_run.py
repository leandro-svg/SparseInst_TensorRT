import json
import sys
import os
import time
from tkinter import N
import numpy as np
import cv2
import onnx
import onnxruntime as ort
from onnx import numpy_helper

img = cv2.imread("/home/nvidia/SSD/Leandro_Intern/SparseInst/skate.jpg")
#img = np.dot(img[...,:3], [0.299,0.587,0.114])
img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_AREA)
img.resize((1,3,640,640))
data = json.dumps({'data':img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
ortvalue = ort.OrtValue.ortvalue_from_numpy(data)
print(ortvalue.device_name())  # 'cpu'
print(ortvalue.shape())        # shape of the numpy array X
print(ortvalue.data_type())    # 'tensor(float)'
print(ortvalue.is_tensor())    # 'True'
print(np.array_equal(ortvalue.numpy(), data))  # 'True'
print(np.shape(data))



onnx_model = onnx.load("/home/nvidia/SSD/Leandro_Intern/SparseInst/output/sparseinst_onnx.onnx")

onnx.checker.check_model(onnx_model)

sess = ort.InferenceSession("/home/nvidia/SSD/Leandro_Intern/SparseInst/output/sparseinst_onnx.onnx")

input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

#########################
# Let's see the output name and shape.

output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)



outputs = sess.run(None, {'input_image': ortvalue})
print(len(outputs))
print(len(outputs[0]))
print(len(outputs[1]))
print(outputs)
print(ortvalue)
print("data", data)
print("img", img)

