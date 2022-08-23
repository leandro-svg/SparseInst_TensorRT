import onnx
import tensorrt as trt
import os
import argparse



def build_engine( onnx_file_path, engine_file_path, flop=16):
    network_flags = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        network_flags
    )
    parser = trt.OnnxParser(network, trt_logger)


    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file')
            for error in range(parser.num_errors):
                print(parser.num_errors)
                print(parser.get_error(error))
            return None

    print("Completed parsing ONNX file")
    builder.max_batch_size = 1

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("cannot removing existing file: ",
            engine_file_path)
    print("Creating Tensorrt Engine")

    config = builder.create_builder_config()
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    config.max_workspace_size = 2 << 30
    config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)

    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Serialized Engine Saved at: ", engine_file_path)
    return engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export model to the onnx format")
    parser.add_argument(
        "--onnx_model",
        default="onnx/sparseinst_giam_onnx_2b7d68_classes_lujzz_without_interpolate_torch2trt_.onnx",
        metavar="FILE",
        help="path to onnx model file",
    )
    parser.add_argument(
        "--output",
        default='engine/sparseinst_giam_onnx_2b7d68_classes_lujzz_without_interpolate_torch2trt_.engine',
        metavar="FILE",
        help="path to the output tensorrt file",
    )

    args = parser.parse_args()
    build_engine(args.onnx_model, args.output)
