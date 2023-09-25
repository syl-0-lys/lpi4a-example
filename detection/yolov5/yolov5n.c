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
    1 * 3 * 384 * 640,
};
const char model_name[] = "network";

#define RESIZE_HEIGHT 384
#define RESIZE_WIDTH 640
#define CROP_HEGHT 384
#define CROP_WIDTH 640
#define R_MEAN 0.0
#define G_MEAN 0.0
#define B_MEAN 0.0
#define SCALE 0.003921568627

static void postprocess_opt(void *sess)
{
    int output_num;
    output_num = csinn_get_output_number(sess);

    struct csinn_tensor *output_tensors[3];

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

    struct shl_yolov5_box out[32];

    const float conf_thres = 0.25f;
    const float iou_thres = 0.45f;
    struct shl_yolov5_params *params = shl_mem_alloc(sizeof(struct shl_yolov5_params));
    params->conf_thres = conf_thres;
    params->iou_thres = iou_thres;
    params->strides[0] = 8;
    params->strides[1] = 16;
    params->strides[2] = 32;
    float anchors[18] = {10.f, 13.f, 16.f, 30.f, 33.f, 23.f,
                         30.f, 61.f, 62.f, 45.f, 59.f, 119.f,
                         116.f, 90.f, 156.f, 198.f, 373.f, 326.f};
    memcpy(params->anchors, anchors, sizeof(anchors));

    int num;
    num = shl_c920_detect_yolov5_postprocess(output_tensors, out, params);
    int i = 0;

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
    int output_num = 3;
    int input_group_num = 1;
    int i;

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
    input_tensors[0]->dim[2] = 384;
    input_tensors[0]->dim[3] = 640;

    float *inputf[input_num];
    int8_t *input[input_num];

    void *input_aligned[input_num];
    for (i = 0; i < input_num; i++)
    {
        input_size[i] = csinn_tensor_byte_size(((struct csinn_session *)sess)->input[i]);
        input_aligned[i] = shl_mem_alloc_aligned(input_size[i], 0);
    }

    uint64_t start_time, end_time;
    for (i = 0; i < input_group_num; i++)
    {
        /* set input */
        for (int j = 0; j < input_num; j++)
        {
            int input_len = csinn_tensor_size(((struct csinn_session *)sess)->input[j]);
            struct image_data *img = get_input_data(data_path[i * input_num + j], input_len);

            inputf[j] = img->data;
            free_image_data(img);

            input[j] = shl_ref_f32_to_input_dtype(j, inputf[j], sess);
        }
        memcpy(input_aligned[0], input[0], input_size[0]);
        input_tensors[0]->data = input_aligned[0];

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
