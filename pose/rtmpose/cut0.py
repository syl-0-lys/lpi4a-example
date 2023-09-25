import onnx

input_path = "rtmpose.onnx"
output_path = "rtmpose0.onnx"
input_names = ["input"]
output_names = ["/final_layer/Conv_output_0"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
