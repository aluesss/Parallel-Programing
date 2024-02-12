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

#include "kernels.h"

/* This is your own kernel, so you should decide which parameters to 
   add here*/
void run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *time_returned) {
  // Figure out how to split the work into threads and call the kernel below.
  int num_pixel = width * height;
  int32_t num_thread = 0;
  if(num_pixel < 1024){
    num_thread = num_pixel;
  }
  else{
    num_thread = 1024;
  }
  
  dim3 threads(num_thread, 1);
  dim3 blocks((int) ceil(num_pixel / (float) num_thread));

  float apply_kernal, apply_normalize, load_kernal, load_normalize, out_normalize;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start); // Collect load_kernal

  int32_t *h_input; 
  int32_t *h_output;
 
  int8_t *h_filter;

  // Malloc Device memory 
  cudaMalloc( (void **) &h_input, num_pixel * sizeof(int32_t));
  cudaMalloc( (void **) &h_output, num_pixel * sizeof(int32_t));
  cudaMalloc( (void **) &h_filter, dimension * dimension * sizeof(int8_t));
  
  // Write to device memory
  cudaMemcpyAsync(h_input, input, num_pixel * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(h_filter , filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&load_kernal, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  cudaEventCreate(&start); // Collect apply_kernal
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //Launch device function with assigned threads and blocks
  kernel5<<<blocks, threads>>> (h_filter, dimension, (const int32_t *) h_input, h_output, width, height);
  
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
  cudaMalloc( &global_max, width * height * sizeof(int32_t));
  cudaMalloc( &global_min, width * height * sizeof(int32_t));
  int iteration_num = width * height;
  int32_t numThreads, numBlocks;
  bool should_repeat = calculate_blocks_and_threads(iteration_num, numBlocks, numThreads);
  gpu_switch_threads(iteration_num, numThreads, numBlocks, h_output, global_min, global_max, 1);
  while(should_repeat){
    iteration_num = numBlocks;
    should_repeat = calculate_blocks_and_threads(iteration_num, numBlocks, numThreads);
    gpu_switch_threads(iteration_num, numThreads, numBlocks, h_output, global_min, global_max, 0);
  }

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&load_normalize, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
  
  cudaEventCreate(&start); // Collect apply_normalize
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Launch Normalized function
  normalize5<<<blocks, threads>>> (h_output, width, height, global_min , global_max);

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&apply_normalize, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
  
  //Free memory
  cudaFree( global_max);
  cudaFree( global_min);

  cudaEventCreate(&start); // Collect out_normalize
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Write Normalization to output
  cudaMemcpyAsync(output, h_output, num_pixel * sizeof(int32_t), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&out_normalize, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  //Free memory   
  cudaFree( h_filter);
  cudaFree( h_input);
  cudaFree( h_output);

  // Write time information
  float time_gpu_computation = apply_kernal + apply_normalize + load_normalize; 
  float time_gpu_transfer_in = load_kernal; 
  float time_gpu_transfer_out = out_normalize;

  time_returned[0] = time_gpu_computation;
  time_returned[1] = time_gpu_transfer_in;
  time_returned[2] = time_gpu_transfer_out; 
}

__global__ void kernel5(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height) {
         //Get thread Id: 
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int32_t row = idx / width; 
      int32_t column = idx % width; 
      //Get pixel to process
      int32_t target = apply2d(filter, input, dimension, width, height, row , column);
      output[idx] = target; 

    //synchronize the local threads writing to the local memory cache
    __syncthreads();
}

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < width * height){
    if (smallest[0]  == biggest[0] ) {
      return;
    }
    __syncthreads();
    image[idx] =
        ((image[idx] - smallest[0]) * 255) / (biggest[0]  - smallest[0]);
  }
}
