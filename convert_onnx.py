import math
import argparse
import  cv2
import torch

from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from asyncio import streams
import os
from detectron2.data.detection_utils import read_image
import torch
from sparseinst import add_sparse_inst_config
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
import argparse

from sparseinst.config import add_sparse_inst_config


def normalizer(x, mean, std): return (x - mean) / std

def main():
    parser = argparse.ArgumentParser(
        description="Export model to the onnx format")
    parser.add_argument(
        "--config-file",
        default="configs/sparse_inst_r50_giam.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=640, type=int)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument(
        "--output",
        default="onnx/sparseinst_giam_onnx_2b7d68_classes_lujzz_without_interpolate_torch2trt_.onnx",
        metavar="FILE",
        help="path to the output onnx file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'weights/sparse_inst_r50_giam_aug_2b7d68.pth'],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--image",
        default='input/input_image/640x640.jpg',
        metavar="FILE",
        help="path to the output onnx file",
    )

     

    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # norm for ONNX: change FrozenBN back to BN
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.NORM = "BN"

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger(output=output_dir)
    logger.info(cfg)


    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    logger.info("Model:\n{}".format(model))

    checkpointer = DetectionCheckpointer(model)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)
    logger.info("load Model:\n{}".format(cfg.MODEL.WEIGHTS))
    device = torch.device('cuda:0')

    input_names = ["input_image"]

    #dummy_input = torch.rand((3, height, width)).to(cfg.MODEL.DEVICE)
    pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.120, 57.375]).to(device).view(3, 1, 1)

    path = args.image
    original_image = read_image(path, format="RGB")
    print(original_image.shape)
    image = cv2.resize(original_image, (640, 640))
    height, width = image.shape[:2]

    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
    image = normalizer(image, pixel_mean, pixel_std)
    image = image.repeat(1,1,1,1)
    print(image.shape)
    dummy_input = image

    output_names = ["scores", "classes", "masks"]

    model.forward = model.forward_test_3
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=False,
        opset_version=11,
    )
    
    logger.info("Done. The onnx model is saved into {}.".format(args.output))


if __name__ == "__main__":
    main()
