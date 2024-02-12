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

void run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *time_returned) {
  // Figure out how to split the work into threads and call the kernel below.

  int num_pixel = width * height;
  int32_t num_thread = 0;
  if(height < 1024){
    num_thread = height; 
  }
  else{
    num_thread = 1024; 
  }
 
  dim3 threads( num_thread , 1); 
  //Only need one block. 
  dim3 blocks(1);
 
  // Timer initialization code 
  float apply_kernal, apply_normalize, load_kernal, load_normalize, out_normalize;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start); // Collect load_kernal

  int32_t *cuda_input;
  int32_t *cuda_output; 
  int8_t *cuda_filter;
  void **p_input =(void **) &cuda_input;
  void **p_output =(void **) &cuda_output;
  void **p_filter =(void **) &cuda_filter;

  // Malloc Device memory 
  cudaMalloc(p_input, num_pixel * sizeof(int32_t));
  cudaMalloc(p_output, num_pixel * sizeof(int32_t));
  cudaMalloc(p_filter, dimension * dimension * sizeof(int8_t));

  // Write to device memory
  cudaMemcpy(cuda_input, input, num_pixel * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_filter , filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&load_kernal, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start); // Collect apply_kernal

  //Launch device function with assigned threads and blocks 
  kernel3<<<blocks, threads>>> (cuda_filter, dimension, (const int32_t *) cuda_input, cuda_output, width, height, (int32_t) ceil(height / 1024.0));
  
  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&apply_kernal, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Start Reduction
  cudaEventRecord(start); // Collect load_normalize
  
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

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start); // Collect apply_normalize

  // Launch Normalized function
  normalize3<<<blocks, threads>>> (cuda_output, width, height, global_min , global_max, (int32_t) ceil(height / 1024.0));

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&apply_normalize, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start); // Collect out_normalize

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

__global__ void kernel3(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t rows_per_thread) 
  {
    int idx = threadIdx.x;
    // Get assigned row 
    int32_t start_row = idx * rows_per_thread; 
    
    for (int i = 0; i < rows_per_thread; i++){ // For each assigned row
      int32_t row = start_row + i;
      if (row < height){
        for (int col = 0; col < width; col++){
          int32_t target = apply2d(filter, input, dimension, width, height, row, col);
          output[(row * width) + col] = target;
        }
      }
    }
    
    //synchronize the local threads writing to the local memory cache
    __syncthreads();
  }

__global__ void normalize3(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest, int32_t rows_per_thread) 
    {
      int idx = threadIdx.x;
      // Get assigned row 
      int32_t start_row = idx * rows_per_thread; 
      
      for (int i = 0; i < rows_per_thread; i++){ // For each assigned row
        int32_t row = start_row + i;
        if (row < height){
          for (int col = 0; col < width; col++){
            if (smallest[0]  == biggest[0] ) {
              return;
            }
            image[(row * width) + col] = ((image[(row * width) + col] - smallest[0]) * 255) / (biggest[0]  - smallest[0]);
          }
        }
      }
      //synchronize the local threads writing to the local memory cache
      __syncthreads();

  }
