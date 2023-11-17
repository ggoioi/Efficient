import tensorrt as trt
import common

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(common.EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30  # 1GB
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        return builder.build_cuda_engine(network)

def infer_with_engine(auto, stu, tea, data_path):
    pass
auto_trt_engine = build_engine('auto_encoder.onnx')
stu_trt_engine = build_engine('student.onnx')
tea_trt_engine = build_engine('teacher.onnx')
if __name__ == '__main__':
