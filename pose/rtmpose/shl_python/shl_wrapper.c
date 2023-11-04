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

#include <libgen.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "io.h"
#include "shl_ref.h"
//#include "shl_thead_rvv.h"
void shl_rvv_f32_to_i16(const float *input, int16_t *output, int32_t offset, float *scale, uint32_t length);
void shl_rvv_i16_to_f32(const int16_t *input, float *output, int32_t offset, float *scale, uint32_t length);

void* csinn_(char* params);
void csinn_update_input_and_run(struct csinn_tensor** input_tensors, void* sess);

void* create_graph(char* params_path) {
  int binary_size;
  char* params = get_binary_from_file(params_path, &binary_size);
  if (params == NULL) {
    return NULL;
  }

  char* suffix = params_path + (strlen(params_path) - 3);
  if (strcmp(suffix, ".bm") == 0) {
      struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
      if (section->graph_offset) {
          return csinn_import_binary_model(params);
      } else {
          return csinn_(params + section->params_offset * 4096);
      }
  } else {
      return NULL;
  }
}

void* load_model(char* model_path) {
  void* sess = create_graph(model_path);
  return sess;
}

void session_run(void* sess, float** data, int num) {
  struct csinn_tensor* input_tensors[num];

  int8_t* input[num];
  for (int i = 0; i < num; i++) {
    struct csinn_tensor* input_t = csinn_alloc_tensor(NULL);
    csinn_get_input(i, input_t, sess);

    input[i] = shl_mem_alloc(csinn_tensor_byte_size(input_t));
    // input[i] = shl_ref_f32_to_input_dtype(i, data[i], sess);
    shl_rvv_f32_to_i16(data[i], input[i], 0, &input_t->qinfo->scale, csinn_tensor_size(input_t));
    // printf("%x, %x, %x, %x, %f, %f, %f, %f\n", input[i][0], input[i][1],input[i][2],input[i][3], data[i][0], data[i][1], data[i][2], data[i][3]);
    input_t->data = input[i];

    input_tensors[i] = input_t;
  }

  csinn_update_input_and_run(input_tensors, sess);

  // free resources
  for (int i = 0; i < num; i++) {
    shl_mem_free(input[i]);
    csinn_free_tensor(input_tensors[i]);
  }
}

void get_output_by_index(void* sess, int index, float* out_data) {
  struct csinn_tensor* output = csinn_alloc_tensor(NULL);
  csinn_get_output(index, output, sess);
  // struct csinn_tensor* foutput = shl_ref_tensor_transform_f32(output);

  // memcpy(out_data, foutput->data, csinn_tensor_byte_size(foutput));
  // shl_ref_tensor_transform_free_f32(foutput);
  shl_rvv_i16_to_f32(output->data, out_data, 0, &output->qinfo->scale, csinn_tensor_size(output));
//   #_hhb_wrapper_free_output_buf_#
  csinn_free_tensor(output);
}
