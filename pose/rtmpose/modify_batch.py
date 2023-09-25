import onnx
model = onnx.load_model("end2end.onnx")
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "1"
onnx.save(model, "rtmpose.onnx")
