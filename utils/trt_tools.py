# 先将onnx模型-->trt模型
# cd E:\Program Files\TensorRT-8.4.3.1\bin
# trtexec --onnx=E:\gitlab\cars_detection\yolov5\predict_result\onnx_model\model.onnx --saveEngine=E:\gitlab\cars_detection\yolov5\predict_result\model.trt --workspace=6000

import tensorrt as trt
import torch

# trt操作部分
def trt_version():
    return trt.__version__


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine创建执行context
            self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        # 创建输出tensor，并分配内存
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)  # 通过binding_name找到对应的input_id
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))  # 找到对应的数据类型
            shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))  # 找到对应的形状大小
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()  # 绑定输出数据指针

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[0].contiguous().data_ptr()  # 应当为inputs[i]，对应3个输入。但由于我们使用的是单张图片，所以将3个输入全设置为相同的图片。

        self.context.execute_async(
            batch_size, bindings, torch.cuda.current_stream().cuda_stream
        )  # 执行推理

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs