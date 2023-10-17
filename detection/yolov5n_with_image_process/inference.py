import numpy as np
import cv2
import os
input_hight = 384
input_width = 640
# 打开默认的摄像头（设备编号为 0）
cap = cv2.VideoCapture(0)
def image_preprocess(image, target_size):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized

    return image_padded
def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def draw_bbox(image, bboxes, classes=read_class_names("coco.names"), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = (255,0,0)
        bbox_thick = 1
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (255, 255, 255), bbox_thick, lineType=cv2.LINE_AA)

    return image
model_image_process_command = "./yolov5n_image_process"
# model_inference_command = "./yolov5n_example ./hhb_out/hhb.bm image_preprocessed.bin"#第一次运行使用这个命令
model_inference_command = "./yolov5n_example ./shl.hhb.bm image_preprocessed_wxw.bin"#除了第一次后面都运行使用这个命令


while True:
    # 从摄像头读取一帧数据
    ret, frame = cap.read()
    success = cv2.imwrite('hat.jpg', frame)
    if success:
        os.system(model_image_process_command)
        os.system(model_inference_command)
    bboxes = []
    with open("detect.txt", 'r') as f:
        x_min = f.readline().strip()
        while x_min:
            y_min = f.readline().strip()
            x_max = f.readline().strip()
            y_max = f.readline().strip()
            probability = f.readline().strip()
            cls_id = f.readline().strip()
            bbox = [float(x_min), float(y_min), float(x_max), float(y_max), float(probability), int(cls_id)]
            print(bbox)
            bboxes.append(bbox)
            x_min = f.readline().strip()
    image_data = image_preprocess(np.copy(frame), [input_hight, input_width])
    image = draw_bbox(image_data, bboxes)
    success1 = cv2.imwrite("hat_result.jpg", image)
    if success1:
        image_show = cv2.imread('hat_result.jpg')
        cv2.imshow('Image', image_show)
    if cv2.waitKey(1) == 27:
        break
   
 
# 释放资源
cap.release()
cv2.destroyAllWindows()