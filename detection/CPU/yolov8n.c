/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/* auto generate by HHB_VERSION "2.3.0" */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <libgen.h>
#include <unistd.h>
#include "io.h"
#include "shl_c920.h"
#include "process.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define FILE_LENGTH 1028
#define SHAPE_LENGHT 128

void *csinn_(char *params);
void csinn_update_input_and_run(struct csinn_tensor **input_tensors, void *sess);
void *csinn_nbg(const char *nbg_file_name);

int input_size[] = {
    1 * 3 * 640 * 640,
};
const char model_name[] = "network";

#define RESIZE_HEIGHT 640
#define RESIZE_WIDTH 640
#define CROP_HEGHT 640
#define CROP_WIDTH 640
#define R_MEAN 0.0
#define G_MEAN 0.0
#define B_MEAN 0.0
#define SCALE 0.003921568627


/** YOLOv8 detect box */
struct shl_yolov8_box {
    int label;   /**< Object label */
    float score; /**< Object confidence */
    float x1;    /**< X1 coordinate of object detection rectangle */
    float y1;    /**< Y1 coordinate of object detection rectangle */
    float x2;    /**< X2 coordinate of object detection rectangle */
    float y2;    /**< Y2 coordinate of object detection rectangle */
    float area;  /**< Area of object detection rectangle */
};


/** YOLOv8 detect params */
struct shl_yolov8_params {
    float conf_thres;   /**< Confidence threshold, must be between 0 and 1 */
    float iou_thres;    /**< IoU threshold for NMS calculation */
};


static void qsort_desc_fp32(int32_t *box_idx, float *scores, int left, int right)
{
    int i = left;
    int j = right;
    float p = scores[(left + right) / 2];
    while (i <= j) {
        while (scores[i] > p) {
            i++;
        }
        while (scores[j] < p) {
            j--;
        }
        if (i <= j) {
            int32_t tmp_idx = box_idx[i];
            box_idx[i] = box_idx[j];
            box_idx[j] = tmp_idx;
            float tmp_score = scores[i];
            scores[i] = scores[j];
            scores[j] = tmp_score;
            i++;
            j--;
        }
    }
    if (j > left) {
        qsort_desc_fp32(box_idx, scores, left, j);
    }
    if (i < right) {
        qsort_desc_fp32(box_idx, scores, i, right);
    }
}


static float get_iou_fp32(const struct shl_yolov8_box box1, const struct shl_yolov8_box box2)
{
    float x1 = fmax(box1.x1, box2.x1);
    float y1 = fmax(box1.y1, box2.y1);
    float x2 = fmin(box1.x2, box2.x2);
    float y2 = fmin(box1.y2, box2.y2);
    float inter_area = fmax(0, x2 - x1) * fmax(0, y2 - y1);
    float iou = inter_area / (box1.area + box2.area - inter_area);
    return iou;
}


static int non_max_suppression_fp32(struct shl_yolov8_box *boxes, int32_t *indices, float iou_thres,
                                    int box_num)
{
    float *scores = (float *)shl_mem_alloc(box_num * sizeof(float));
    int32_t *box_indices = (int32_t *)shl_mem_alloc(box_num * sizeof(int32_t));
    for (int i = 0; i < box_num; i++) {
        scores[i] = boxes[i].score;
        box_indices[i] = i;
    }
    qsort_desc_fp32(box_indices, scores, 0, box_num - 1);

    int box_cnt = 0;
    for (int i = 0; i < box_num; i++) {
        bool keep = true;
        int32_t box_idx = box_indices[i];
        struct shl_yolov8_box box1 = boxes[box_idx];
        for (int j = 0; j < box_cnt; j++) {
            struct shl_yolov8_box box2 = boxes[indices[j]];
            float iou = get_iou_fp32(box1, box2);
            if (iou > iou_thres) {
                keep = false;
            }
        }
        if (keep) {
            indices[box_cnt++] = box_idx;
        }
    }

    shl_mem_free(box_indices);
    shl_mem_free(scores);

    return box_cnt;
}

static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

int max(float* src, int len)
{
    float max;
    float val;
    int i, num;

    max = src[0];
    for(i =0; i<len ; i++){
        val = *(src + i);
        if(val-max > 0){
            max = val;
        }
        else{
            max = max;
        }
    }
    return max;
}


static void proposal_fp32(struct csinn_tensor *input,float conf_thres, struct shl_yolov8_box *box, 
                          int *box_num)
{
//8400*6 -> 8400*7
    float *data = (float *)input->data;
    const int inner_size = input->dim[2]; //xywh + cls
    const int cls_num = inner_size - 4; 
    const int grid_size = input->dim[1];

/* sigmoid(x) > t  <=>  x > -ln(1/t-1) */
    float threshold = -log(1.f / conf_thres - 1.f);

    float *cls_confs = (float *)shl_mem_alloc(cls_num * sizeof(float));
    float *feat = (float *)shl_mem_alloc(inner_size * sizeof(float));

printf("proposal_start \n");
printf("inner_size: %d, cls_num: %d \n", inner_size, cls_num);

    for (int q = 0; q < grid_size; q++) {
        for(int i=0; i<inner_size; i++){
           *(feat + i) = *(data + q * inner_size + i); // xywh + cls
        }

//xywh + conf + cls
        for(int i=0; i<cls_num; i++){
           cls_confs[i] = *(feat + (4+i));
        }

        float box_score = max(cls_confs, cls_num);

        if (box_score <= threshold) {
            continue;
        }

        float max_score = -FLT_MAX;
        int max_idx = -1;
        for (int k = 4; k < inner_size; k++) {
            float score = feat[k];
            if (score > max_score) {
                max_score = score;
                max_idx = k - 4;
            }
        }


        float box_conf = sigmoid(box_score);
        float class_conf = box_conf * sigmoid(max_score);
        if (class_conf <= conf_thres) {
            continue;
        }

        // float dx = sigmoid(feat[0]);
        // float dy = sigmoid(feat[1]);
        // float dw = sigmoid(feat[2]);
        // float dh = sigmoid(feat[3]);

        float pb_cx = feat[0];
        float pb_cy = feat[1];

        float pb_w = feat[2];
        float pb_h = feat[3];

        box[*box_num].x1 = pb_cx - pb_w * 0.5f;
        box[*box_num].y1 = pb_cy - pb_h * 0.5f;
        box[*box_num].x2 = pb_cx + pb_w * 0.5f;
        box[*box_num].y2 = pb_cy + pb_h * 0.5f;
        box[*box_num].label = max_idx;
        box[*box_num].score = max_score;
        box[*box_num].area =
            (box[*box_num].x2 - box[*box_num].x1) * (box[*box_num].y2 - box[*box_num].y1);
        *box_num += 1;

        //     }
        // }
    }

    shl_mem_free(feat);
    shl_mem_free(cls_confs);
}

/*
 * Preprocess function
 */
void preprocess(struct image_data *img, int is_rgb, int to_bgr)
{
    uint32_t new_height, new_width;
    uint32_t min_side;
    if (is_rgb) {
        im2rgb(img);
    }
    if (RESIZE_WIDTH == 0) {
        min_side = MIN(img->shape[0], img->shape[1]);
        new_height = (uint32_t) (img->shape[0] * (((float)RESIZE_HEIGHT) / (float)min_side));
        new_width = (uint32_t) (img->shape[1] * (((float)RESIZE_HEIGHT) / (float)min_side));
        imresize(img, new_height, new_width);
    } else {
        imresize(img, RESIZE_HEIGHT, RESIZE_WIDTH);
    }
    data_crop(img, CROP_HEGHT, CROP_WIDTH);
    sub_mean(img, R_MEAN, G_MEAN, B_MEAN);
    data_scale(img, SCALE);
    if(to_bgr) {
        imrgb2bgr(img);
    }
    imhwc2chw(img);
}

int detect_yolov8_postprocess(struct csinn_tensor **input_tensors,
                                       struct shl_yolov8_box *out, struct shl_yolov8_params *params)
{
    /* [1, 255, x, y] */
    struct csinn_tensor *input0 = input_tensors[0];
    
    printf("11 \n");
    if (!((input0->dtype == CSINN_DTYPE_FLOAT32) || (input0->dtype == CSINN_DTYPE_UINT8)))
    {
        shl_debug_error("yolov8 posprocess unsupported dtype: %d", input0->dtype);
        return 0;
    }

    const int max_box = input0->dim[1]; //

    const float conf_thres = params->conf_thres;
    const float iou_thres = params->iou_thres;

    if (!(conf_thres > 0.f && conf_thres < 1.f)) {
        shl_debug_error("Confidence threshold must be between 0 and 1!");
        return 0;
    }
printf("12 \n");

    struct shl_yolov8_box proposals[max_box];
    int box_num = 0;

    if (input0->dtype == CSINN_DTYPE_FLOAT32) 
    {
        proposal_fp32(input0, conf_thres, proposals, &box_num);
    } 
    // else if (input0->dtype == CSINN_DTYPE_UINT8) 
    // {
    //     proposal_uint8(input0, conf_thres, proposals, &box_num);
    // }

printf("proposal_done \n");
printf("box_num = %d \n", box_num);

    if (box_num == 0) {
        return 0;
    }

    int32_t *indices = (int32_t *)shl_mem_alloc(box_num * sizeof(int32_t));
    int num = non_max_suppression_fp32(proposals, indices, iou_thres, box_num);

printf("nms \n");
printf("nms_OP_num = %d \n", num);

    for (int i = 0; i < num; i++) {
        int idx = indices[i];
        out[i].label = proposals[idx].label;
        out[i].score = proposals[idx].score;
        out[i].x1 = proposals[idx].x1;
        out[i].y1 = proposals[idx].y1;
        out[i].x2 = proposals[idx].x2;
        out[i].y2 = proposals[idx].y2;
    }
printf("post_done \n");
    shl_mem_free(indices);
    return num;
}


static void postprocess_opt(void *sess)
{
    int output_num;
    output_num = csinn_get_output_number(sess);

    struct csinn_tensor *output_tensors[output_num];

printf("1 \n");

    struct csinn_tensor *output;
    for (int i = 0; i < output_num; i++)
    {
        output = csinn_alloc_tensor(NULL);
        output->data = NULL;
        csinn_get_output(i, output, sess);

        output_tensors[i] = output;

        struct csinn_tensor *ret = csinn_alloc_tensor(NULL);
        csinn_tensor_copy(ret, output);
        if (ret->qinfo != NULL)
        {
            shl_mem_free(ret->qinfo);
            ret->qinfo = NULL;
        }
        ret->quant_channel = 0;
        ret->dtype = CSINN_DTYPE_FLOAT32;
        ret->data = shl_c920_output_to_f32_dtype(i, output->data, sess);
        output_tensors[i] = ret;
    }
printf("2 \n");
    struct shl_yolov8_box out[90];

    const float conf_thres = 0.30f;
    const float iou_thres = 0.45f;
    struct shl_yolov8_params *params = shl_mem_alloc(sizeof(struct shl_yolov8_params));
    params->conf_thres = conf_thres;
    params->iou_thres = iou_thres;
    // params->strides[0] = 8;
    // params->strides[1] = 16;
    // params->strides[2] = 32;
    // float anchors[18] = {10.f, 13.f, 16.f, 30.f, 33.f, 23.f,
    //                      30.f, 61.f, 62.f, 45.f, 59.f, 119.f,
    //                      116.f, 90.f, 156.f, 198.f, 373.f, 326.f};
    // memcpy(params->anchors, anchors, sizeof(anchors));
printf("3 \n");
    int num;
    num = detect_yolov8_postprocess(output_tensors, out, params);
    int i = 0;

printf("4 \n");
    FILE *fp = fopen("detect.txt", "w+");

    printf("detect num: %d\n", num);
    printf("id:\tlabel\tscore\t\tx1\t\ty1\t\tx2\t\ty2\n");
    for (int k = 0; k < num; k++)
    {
        printf("[%d]:\t%d\t%f\t%f\t%f\t%f\t%f\n", k, out[k].label,
               out[k].score, out[k].x1, out[k].y1, out[k].x2, out[k].y2);
        fprintf(fp, "%f\n%f\n%f\n%f\n%f\n%d\n",
                out[k].x1, out[k].y1, out[k].x2, out[k].y2, out[k].score, out[k].label);
    }
    printf("5 \n");
    fclose(fp);

    shl_mem_free(params);

    for (int i = 0; i < 3; i++)
    {
        csinn_free_tensor(output_tensors[i]);
    }
    csinn_free_tensor(output);
}


void *create_graph(char *params_path)
{
    int binary_size;
    char *params = get_binary_from_file(params_path, &binary_size);
    if (params == NULL)
    {
        return NULL;
    }

    char *suffix = params_path + (strlen(params_path) - 7);
    if (strcmp(suffix, ".params") == 0)
    {
        // create general graph
        return csinn_(params);
    }

    suffix = params_path + (strlen(params_path) - 3);
    if (strcmp(suffix, ".bm") == 0)
    {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        if (section->graph_offset)
        {
            return csinn_import_binary_model(params);
        }
        else
        {
            return csinn_(params + section->params_offset * 4096);
        }
    }
    else
    {
        return NULL;
    }
}


int main(int argc, char **argv)
{
    char **data_path = NULL;
    int input_num = 1;
    int input_group_num = 1;
    
    int i, j, k;
    uint64_t start_time, end_time;

    if (argc < (2 + input_num))
    {
        printf("Please set valide args: ./model.elf model.params "
               "[tensor1/image1 ...] [tensor2/image2 ...]\n");
        return -1;
    }
    else
    {
        if (argc == 3 && get_file_type(argv[2]) == FILE_TXT)
        {
            data_path = read_string_from_file(argv[2], &input_group_num);
            input_group_num /= input_num;
        }
        else
        {
            data_path = argv + 2;
            input_group_num = (argc - 2) / input_num;
        }
    }

    void *sess = create_graph(argv[1]);

    struct csinn_tensor *input_tensors[input_num];
    input_tensors[0] = csinn_alloc_tensor(NULL);
    input_tensors[0]->dim_count = 4;
    input_tensors[0]->dim[0] = 1;
    input_tensors[0]->dim[1] = 3;
    input_tensors[0]->dim[2] = 640;
    input_tensors[0]->dim[3] = 640;

    float *inputf[input_num];
    int8_t *input[input_num];
    void *input_aligned[input_num];

    for (i = 0; i < input_num; i++)
    {
        input_size[i] = csinn_tensor_byte_size(((struct csinn_session *)sess)->input[i]);
        input_aligned[i] = shl_mem_alloc_aligned(input_size[i], 0);
    }


    for (i = 0; i < input_group_num; i++)
    {
/* set input */
        for (j = 0; j < input_num; j++)
        {
            /*if (get_file_type(data_path[i * input_num + j]) != FILE_BIN) {
                printf("Please input binary files, since you compiled the model without preprocess.\n");
                return -1;
            }*/
            int input_len = csinn_tensor_size(((struct csinn_session *)sess)->input[j]);
            struct image_data *img = get_input_data(data_path[i * input_num + j], input_len);
            if (get_file_type(data_path[i * input_num + j]) == FILE_PNG || get_file_type(data_path[i * input_num + j]) == FILE_JPEG) {
                preprocess(img, 1, 0);
            }
            inputf[j] = img->data;
            free_image_data(img);
            
            //inputf[j] = (float*)get_binary_from_file(data_path[i * input_num + j], NULL);

            input[j] = shl_ref_f32_to_input_dtype(j, inputf[j], sess);

            memcpy(input_aligned[j], input[j], input_size[j]);
            input_tensors[j]->data = input_aligned[j];
        }

//run
        start_time = shl_get_timespec();
        
        csinn_update_input_and_run(input_tensors, sess);

        end_time = shl_get_timespec();
        printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time - start_time)) / 1000000,
               1000000000.0 / ((float)(end_time - start_time)));

        postprocess_opt(sess);

        for (int j = 0; j < input_num; j++)
        {
            shl_mem_free(inputf[j]);
            shl_mem_free(input[j]);
        }
    }

    for (int j = 0; j < input_num; j++)
    {
        csinn_free_tensor(input_tensors[j]);
        shl_mem_free(input_aligned[j]);
    }

    csinn_session_deinit(sess);
    csinn_free_session(sess);

    return 0;
}
