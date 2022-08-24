import os
import time
from tkinter import N
import onnxruntime
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.data.detection_utils import read_image
import torch
import detectron2.data.transforms as T
from sparseinst import add_sparse_inst_config
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
import argparse
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import cv2 as cv
from detectron2.structures import Instances, BitMasks
import onnx
import torch.nn as nn
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_numpy_data():
    batch_size = 1
    img_input = np.ones((1,640,640), dtype=np.float32)
    return img_input

def normalizer(x, mean, std): return (x - mean) / std

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def _load_engine(engine_file_path):
    trt_logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_file_path, 'rb') as f:
        with trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            print('_load_engine ok.')
    return engine

def _allocate_buffer(engine):
    binding_names=[]
    for idx in range(100):
        bn= engine.get_binding_name(idx)
        if bn:
            binding_names.append(bn)
        else:
            break
    
    inputs = []
    outputs  = []
    bindings = [None]*len(binding_names)
    stream = cuda.Stream()

    for binding in binding_names:
        binding_idx = engine[binding]
        if binding_idx == -1:
            print("Error Binding Names!")
            continue
        size = trt.volume(engine.get_binding_shape(binding))*engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings[binding_idx] = int(device_mem)

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    print('_allocate_buffer ok.')
    return inputs, outputs, bindings, stream

def _test_engine(engine_file_path, data_input, index, num_times = 100):
    engine = _load_engine(engine_file_path)
    inputs_bufs, output_bufs, bindings, stream = _allocate_buffer(engine)

    batch_size = 1
    context = engine.create_execution_context()
    
    inputs_bufs[0].host = data_input
    cuda.memcpy_htod_async(
        inputs_bufs[0].device,
        inputs_bufs[0].host,
        stream
    )
    context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.handle
    )
    cuda.memcpy_dtoh_async(
        output_bufs[0].host,
        output_bufs[0].device,
        stream
    )
    cuda.memcpy_dtoh_async(
        output_bufs[1].host,
        output_bufs[1].device,
        stream
    )
    cuda.memcpy_dtoh_async(
        output_bufs[2].host,
        output_bufs[2].device,
        stream
    )
    stream.synchronize()
    trt_outputs = [output_bufs[0].host.copy(),output_bufs[1].host.copy(),output_bufs[2].host.copy()]
    ##########
    start = time.time()
    for _ in range(num_times):
        inputs_bufs[0].host = data_input
        cuda.memcpy_htod_async(
            inputs_bufs[0].device,
            inputs_bufs[0].host,
            stream
        )
        context.execute_async_v2(
            bindings=bindings,
            stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(
            output_bufs[0].host,
            output_bufs[0].device,
            stream
        )
        cuda.memcpy_dtoh_async(
            output_bufs[1].host,
            output_bufs[1].device,
            stream
        )
        cuda.memcpy_dtoh_async(
            output_bufs[2].host,
            output_bufs[2].device,
            stream
        )
        stream.synchronize()
        trt_outputs = [output_bufs[0].host.copy(), output_bufs[1].host.copy(), output_bufs[2].host.copy()]
    
    end = time.time()
    time_use_trt = end - start
    print(f"TRT use time {(time_use_trt)}for loop {num_times}, FPS={num_times*batch_size/time_use_trt}")
    return trt_outputs


def test_engine(data_input,index, loop = 10):
    engine_file_path = TENSORRT_ENGINE_PATH_PY
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    trt_outputs = None
    try: 
        trt_outputs = _test_engine(engine_file_path, data_input,index, loop)

    finally:
        cuda_ctx.pop()
    return trt_outputs[0], trt_outputs[1], trt_outputs[2]

def get_image(path):
    
    original_image = read_image(path, format="RGB")

    device = torch.device('cuda:0')
    h, w = (640,640)
    image = cv.resize(original_image, (h, w))
    pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.120, 57.375]).to(device).view(3, 1, 1)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
    image = normalizer(image, pixel_mean, pixel_std)
    image = image.repeat(1,1,1,1)
    
    return image, original_image

def post_process(pred_scores, pred_classes, mask_pred_per_image, mask_threshold):
    mask_pred_per_image = mask_pred_per_image.reshape((100,160,160))
    m = nn.UpsamplingBilinear2d(scale_factor=4.0)
    mask_pred_per_image  = torch.tensor(mask_pred_per_image)
    mask_pred = m(mask_pred_per_image.unsqueeze(0)).squeeze(0)
    mask_pred = mask_pred > mask_threshold

    predictions_mask = mask_pred.reshape((100,640,640))
    ori_shape = (1,3,640,640)
    mask_pred = BitMasks(predictions_mask)
    results = []
    result = Instances(ori_shape)
    result.pred_masks = mask_pred
    result.scores = pred_scores
    result.pred_classes = pred_classes
    results.append(result)

    processed_results = [{"instances": r} for r in results]
    predictions = processed_results[0]

    return predictions, result

def post_process_pytorch(predictions):
    predictions["instances"].scores = np.array((predictions["instances"].scores).cpu())
    predictions["instances"].pred_masks = np.array((predictions["instances"].pred_masks).cpu())
    predictions["instances"].pred_classes = np.array((predictions["instances"].pred_classes).cpu())
    predictions["instances"].pred_masks = BitMasks(predictions["instances"].pred_masks)
    return predictions

def test_trt(img_input, loop=10):
    img_input = np.array(img_input.cpu())
    img_input = np.ascontiguousarray(img_input, dtype=np.float32) 
    predictions_score, predictions_class, predictions_mask = test_engine(img_input, 0,loop)  

    predictions, result = post_process(predictions_score, predictions_class, predictions_mask, mask_threshold)
    return predictions_class, predictions_score, predictions_mask, predictions


def test_onnx(image, mask_threshold, loop=10 ):
    
    model = onnx.load(ONNX_SIM_MODEL_PATH)
    onnx.checker.check_model(model) 

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] # 30 ms

    sess = onnxruntime.InferenceSession(ONNX_SIM_MODEL_PATH, providers = providers)
    outputs = [node.name for node in sess.get_outputs()]
    input_onnx = image.cpu().numpy().astype(np.float32)

    batch_size = 1
    time1 = time.time()
    for i in range(loop):
        out_ort_img_class = sess.run([outputs[1]], {sess.get_inputs()[0].name: input_onnx,})
        out_ort_img_scores = sess.run([outputs[0]], {sess.get_inputs()[0].name: input_onnx,})
        out_ort_img_masks = sess.run([outputs[2]], {sess.get_inputs()[0].name: input_onnx,}) 
    time2 = time.time()
    time_use_onnx = time2 - time1
    print(f'ONNX use time {time_use_onnx} for loop {loop}, FPS= {loop*batch_size/time_use_onnx}')
    
    pred_scores = out_ort_img_scores[0][0][:]
    pred_classes= out_ort_img_class[0][0][:]

    predictions, result = post_process(pred_scores, pred_classes, out_ort_img_masks[0], mask_threshold)

    return pred_classes, pred_scores, out_ort_img_masks[0], predictions, result

def test_pytorch(original_image, loop=10):
    with torch.no_grad():
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        time1 = time.time()
        for i in range(loop):
            print("inputs", inputs)
            predictions = model([inputs])[0]
        time2 = time.time()
        print("predictions", predictions)
        time_use_pytorch = time2 - time1
        print(f'Pytorch use time {time_use_pytorch} for loop {loop}, FPS= {loop/time_use_pytorch}')

        predictions = post_process_pytorch(predictions)
        return predictions
    


def demonstration(img, original_image,  predictions, args_output):
    cpu_device = torch.device("cpu")
    visualizer = Visualizer(original_image, metadata,
                                instance_mode=instance_mode)
    instances = predictions["instances"]#.to(cpu_device)
    instances = instances[instances.scores > 0.5]
    predictions["instances"] = instances
    vis_output = visualizer.draw_instance_predictions(
        predictions=instances)
    if args_output:
        if os.path.isdir(args_output):
            assert os.path.isdir(args_output), args_output
            out_filename = os.path.join(
                args_output, os.path.basename(path))
        else:
            assert len(
                args_output) > 0, "Please specify a directory with args_output"
            out_filename = args_output
        vis_output.save(out_filename)
    


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/sparse_inst_r50_giam.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--output_onnx",
        default="results/result_onnx",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--output_tensorRT",
        default="results/result_tensorrt",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--output_pytorch",
        default="results/result_pytorch",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--input",
        default="input/input_image/640x640.jpg",
        help="A file or directory of your input data "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--onnx_engine",
        default='onnx/sparseinst_giam_onnx_2b7d68_classes_lujzz_without_interpolate_torch2trt_.onnx',
        help="A file or directory of your onnx model. ",
    )
    parser.add_argument(
        "--tensorRT_engine",
        default='engine/sparseinst_giam_onnx_2b7d68_classes_lujzz_without_interpolate_torch2trt_.engine',
        help="A file or directory of your tensorRT model. ",
    )


    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    TENSORRT_ENGINE_PATH_PY = args.tensorRT_engine
    ONNX_SIM_MODEL_PATH = args.onnx_engine


    cfg = setup_cfg(args)
    img_format = cfg.INPUT.FORMAT
    model = build_model(cfg)
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    instance_mode = ColorMode.IMAGE
    mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
    logger.info("load Model:\n{}".format(cfg.MODEL.WEIGHTS))
    device = torch.device('cuda:0')
    aug = T.ResizeShortestEdge([640,640], 640)


    path = args.input
    dummy_input = get_numpy_data()
    dummy = True
    if dummy:
        img_input, original_image = get_image(path)
    else:
        img_input = dummy_input



    predictions_class, predictions_score, predictions_mask, predictions = test_trt(img_input, loop=100)
    demonstration(img_input, original_image, predictions, args.output_tensorRT)

    predictions = test_pytorch(original_image, loop=100)
    demonstration(img_input, original_image, predictions, args.output_pytorch)
    
    out_ort_img_class, out_ort_img_scores, out_ort_img_masks, predictions, result = test_onnx(img_input, mask_threshold, loop=100)
    demonstration(img_input, original_image, predictions, args.output_onnx)
    

    keep = out_ort_img_scores > 0.1
    predictions_score = predictions_score[keep]
    out_ort_img_scores = out_ort_img_scores[keep]

    mse = np.square(np.subtract(out_ort_img_scores, predictions_score)).mean()
    print('Score MSE between onnx and trt result: ', mse)
    
