### 有关说明
此文件夹是在yolov5文件夹上进行了修改，在原有基础上讲python 图像预处理修改为了c++图像预处理，因为经过测试python预处理会占用2-3s的时间，修改为c++速度提高了不少，同样是以os.system去执行图像预处理命令。
[实测视频](https://www.bilibili.com/video/BV1BC4y1G7Qo/?spm_id_from=333.999.0.0&vd_source=b1fff0f773136d7d05331087929c7739)

### 新增文件说明
- main.c：为图像预处理
- inference.py：在原来inference.py基础上新加入了usb摄像头实时读取，调用命令行进行图像预处理。

### 教程

#### 1、搭建docker环境并启动容器
首先要在自己的电脑上安装 Docker，先卸载可能存在的 Docker 版本：
```
sudo apt-get remove docker docker-engine docker.io containerd runc
```


安装Docker依赖的基础软件：
```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```

添加官方源
```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```
安装 Docker：
```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```
安装完毕后，获取 HHB 环境的 Docker 镜像


```
docker pull hhb4tools/hhb:2.4.5

```
拉取镜像完毕后，使用下面的命令进入 Docker 镜像：
```
docker run -itd --name=your.hhb2.4 -p 22 "hhb4tools/hhb:2.4.5"
docker exec -it your.hhb2.4 /bin/bash
```
进入 Docker 镜像后，可使用下面的命令确认 HHB 版本并配置交叉编译环境：
```
hhb --version
export PATH=/tools/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-light.1/bin/:$PATH
```


模型量化
```
cd /home/example/th1520_npu/yolov5n
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip3 install ultralytics
python3 export.py --weights yolov5n.pt --include onnx -imgsz 384 640

将lpi4a-example\detection\yolov5n_with_image_process文件夹拷贝到/home/example/th1520_npu/下以及上述导出的onnx模型
cd /home/example/th1520_npu/yolov5n_with_image_process
hhb -D --model-file yolov5n.onnx --data-scale-div 255 --board th1520 --input-name "images" --output-name "/model.24/m.0/Conv_output_0;/model.24/m.1/Conv_output_0;/model.24/m.2/Conv_output_0" --input-shape "1 3 384 640" --calibrate-dataset kite.jpg  --quantization-scheme "int8_asym"
riscv64-unknown-linux-gnu-gcc yolov5n.c -o yolov5n_example hhb_out/io.c hhb_out/model.c -Wl,--gc-sections -O2 -g -mabi=lp64d -I hhb_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ -lshl -L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/decode/install/lib/rv -L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/runtime/riscv_linux -lprebuilt_runtime -ljpeg -lpng -lz -lstdc++ -lm -I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/ -mabi=lp64d -march=rv64gcv0p7_zfh_xtheadc -Wl,-unresolved-symbols=ignore-in-shared-libs
执行上面命令即可得到图像推理可执行文件以及模型文件

cd /home/example/th1520_npu/
git clone https://github.com/zhangwm-pt/prebuilt_opencv.git
cd /home/example/th1520_npu/yolov5n_with_image_process
执行
riscv64-unknown-linux-gnu-g++ main.cpp -I../prebuilt_opencv/include/opencv4 -L../prebuilt_opencv/lib   -lopencv_imgproc   -lopencv_imgcodecs -L../prebuilt_opencv/lib/opencv4/3rdparty/ -llibjpeg-turbo -llibwebp -llibpng -llibtiff -llibopenjp2    -lopencv_core -ldl  -lpthread -lrt -lzlib -lcsi_cv -latomic -static -o yolov5n_image_process
docker容器会报一个warning，不用管，不影响，你将会得到一个图像预处理可执行文件。

```


下面是一些输出结果：
```
输出
INFO: NNA clock:792000 [kHz]
INFO: Heap :ocm (0x18)
INFO: Heap :anonymous (0x2)
INFO: Heap :dmabuf (0x2)
INFO: Heap :unified (0x5)
FATAL: Importing 737280 bytes of CPU memory has failed (Invalid argument)
Run graph execution time: 16.25801ms, FPS=61.51
detect num: 1
id:     label   score           x1              y1              x2              y2
[0]:    0       0.964871        245.258789      110.341064      348.670044      221.488525
[245.258789, 110.341064, 348.670044, 221.488525, 0.964871, 0]
FPS:  3.1019127075279904
INFO: NNA clock:792000 [kHz]
INFO: Heap :ocm (0x18)
INFO: Heap :anonymous (0x2)
INFO: Heap :dmabuf (0x2)
INFO: Heap :unified (0x5)
FATAL: Importing 737280 bytes of CPU memory has failed (Invalid argument)
Run graph execution time: 17.01332ms, FPS=58.78
detect num: 1
id:     label   score           x1              y1              x2              y2
[0]:    0       0.964715        241.052231      122.274635      332.669098      222.602631
[241.052231, 122.274635, 332.669098, 222.602631, 0.964715, 0]
FPS:  2.9606531317543388
INFO: NNA clock:792000 [kHz]
INFO: Heap :ocm (0x18)
INFO: Heap :anonymous (0x2)
INFO: Heap :dmabuf (0x2)
INFO: Heap :unified (0x5)
FATAL: Importing 737280 bytes of CPU memory has failed (Invalid argument)
Run graph execution time: 19.07990ms, FPS=52.41
detect num: 1
id:     label   score           x1              y1              x2              y2
[0]:    0       0.956770        236.068115      108.511169      339.479370      219.658630
[236.068115, 108.511169, 339.47937, 219.65863, 0.95677, 0]
FPS:  2.8175332718903916
INFO: NNA clock:792000 [kHz]
INFO: Heap :ocm (0x18)
INFO: Heap :anonymous (0x2)
INFO: Heap :dmabuf (0x2)
INFO: Heap :unified (0x5)
FATAL: Importing 737280 bytes of CPU memory has failed (Invalid argument)
Run graph execution time: 16.18402ms, FPS=61.79
detect num: 1
id:     label   score           x1              y1              x2              y2
[0]:    0       0.951897        236.068115      107.003906      339.479370      218.151367
[236.068115, 107.003906, 339.47937, 218.151367, 0.951897, 0]
FPS:  2.919535525401738

```
