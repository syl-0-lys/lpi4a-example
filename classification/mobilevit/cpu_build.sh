#!/bin/bash

hhb -D -f mobilevit.onnx  --calibrate-dataset persian_cat.jpg --data-scale 0.003922 --data-mean "0 0 0"  --board c920 -sd persian_cat.jpg --postprocess save_and_top5 -in "input" -on "output" -is "1 3 256 256" --quantization-scheme float16 --use-custom-fusion

OPENCV_DIR=../../modules/opencv/
riscv64-unknown-linux-gnu-g++ main.cpp -I${OPENCV_DIR}/include/opencv4 -L${OPENCV_DIR}/lib   -lopencv_imgproc   -lopencv_imgcodecs -L${OPENCV_DIR}/lib/opencv4/3rdparty/ -llibjpeg-turbo -llibwebp -llibpng -llibtiff -llibopenjp2    -lopencv_core -ldl  -lpthread -lrt -lzlib -lcsi_cv -latomic -static -o mobilevit_example
