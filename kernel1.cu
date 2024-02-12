/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2022 Bogdan Simion
 * -------------
 */
#include <stdio.h>
#include "kernels.h"
#define MY_MIN(x, y) ((x < y) ? x : y)
#define MY_MAX(x, y) ((x > y) ? x : y)

void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *time_returned) {
  // Figure out how to split the work into threads and call the kernel below.


  int num_pixel = width * height;
  int32_t num_thread = MY_MIN(num_pixel, 1024);

  //Determine threads and blocks needed
  dim3 threads(num_thread, 1);
  dim3 blocks((int) ceil(num_pixel / (float) num_thread));
 
   
  // Timer initialization code 
  float apply_kernal, apply_normalize, load_kernal, load_normalize, out_normalize;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start); // Collect load_kernal

  int32_t *cuda_input;
  int32_t *cuda_output; 
  int8_t *cuda_filter;

  // Malloc Device memory 
  cudaMalloc((void **) &cuda_input, num_pixel * sizeof(int32_t));
  cudaMalloc((void **) &cuda_output, num_pixel * sizeof(int32_t));
  cudaMalloc((void **) &cuda_filter, dimension * dimension * sizeof(int8_t));



  // Write to device memory
  cudaMemcpy(cuda_input, input, num_pixel * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_filter , filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);
  
  //Synchronize device. 
  cudaDeviceSynchronize();
  
  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&load_kernal, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  cudaEventCreate(&start); // Collect apply_kernal
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  //Launch device function with assigned threads and blocks 
  kernel1<<<blocks, threads>>> (cuda_filter, dimension, (const int32_t *) cuda_input, cuda_output, width, height);
  
  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&apply_kernal, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
  
  // Start Reduction
  cudaEventCreate(&start); // Collect load_normalize
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  int32_t *global_min;
  int32_t *global_max;
  
  cudaMalloc(&global_max, width * height * sizeof(int32_t));
  cudaMalloc(&global_min, width * height * sizeof(int32_t));
  int iteration_num = width * height;
  int32_t numThreads, numBlocks;
  bool should_repeat = calculate_blocks_and_threads(iteration_num, numBlocks, numThreads);
  gpu_switch_threads(iteration_num, numThreads, numBlocks, cuda_output, global_min, global_max, 1);
  while(should_repeat){
    iteration_num = numBlocks;
    should_repeat = calculate_blocks_and_threads(iteration_num, numBlocks, numThreads);
    gpu_switch_threads(iteration_num, numThreads, numBlocks, cuda_output, global_min, global_max, 0);
  }

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&load_normalize, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
  
  // Start Normalized
  cudaEventCreate(&start); // Collect apply_normalize
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Launch Normalized function
  normalize1<<<blocks, threads>>> (cuda_output, width, height, global_min, global_max);
  
  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&apply_normalize, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
 
  
  cudaEventCreate(&start); // Collect out_normalize
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  // Write Normalization to output
  cudaMemcpy(output, cuda_output, num_pixel * sizeof(int32_t), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&out_normalize, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
  
  //Free memory
  cudaFree(global_max);
  cudaFree(global_min); 
  cudaFree(cuda_input);
  cudaFree(cuda_output); 
  cudaFree(cuda_filter);

  
  // Write time information
  float time_gpu_computation = apply_kernal + apply_normalize + load_normalize; 
  float time_gpu_transfer_in = load_kernal; 
  float time_gpu_transfer_out = out_normalize;

  time_returned[0] = time_gpu_computation;
  time_returned[1] = time_gpu_transfer_in;
  time_returned[2] = time_gpu_transfer_out; 
   
}

__global__ void kernel1(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t row;
  if(height == 1){
    row = 0;
  }
  else{
    row = idx % height;
  }
  int column = idx / height;
  int index = (row * width) + column;
  if(idx < width*height && index < width*height){
    int32_t target = apply2d(filter, input, dimension, width, height, row, column);
    output[index] = target; 
  }
  //synchronize the local threads writing to the local memory cache
  __syncthreads();
}

__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t row;
  if(height == 1){
    row = 0;
  }
  else{
    row = idx % height;
  }
  int column = idx / height;
  int index = (row * width) + column;

  if(index < width * height){
    if (smallest[0] == biggest[0]) {
      return;
    }
    image[index] =
        ((image[index] - smallest[0]) * 255) / (biggest[0] - smallest[0]);
  }


}
