import onnx

input_path = "rtmpose.onnx"
output_path = "rtmpose1.onnx"
input_names = ["/final_layer/Conv_output_0"]
output_names = ["simcc_x", "simcc_y"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
