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

#include <limits.h>
#include <stdio.h>  
#include <stdlib.h>
#include <stdint.h>
#ifndef __KERNELS__H
#define __KERNELS__H
#define MY_MIN(x, y) ((x < y) ? x : y)
#define MY_MAX(x, y) ((x > y) ? x : y)

/* TODO: you may want to change the signature of some or all of those functions,
 * depending on your strategy to compute min/max elements.
 * Be careful: "min" and "max" are names of CUDA library functions
 * unfortunately, so don't use those for variable names.*/

//From A2
__inline__ __device__ int32_t apply2d(const int8_t *f, const int32_t *original, int32_t dimension,
                int32_t width, int32_t height, int row, int column) {
  int total = dimension * dimension; //Total pixel in the filter
  int center_row = dimension / 2, center_column = dimension / 2; //Find the center
  int sum = 0;
  for(int i=0; i<total; i++){
    int current_row = i / dimension, current_column = i % dimension;
    //Compute valid neighbour
    int x = row + (current_row - center_row);
    int y = column + (current_column - center_column);
    if(x >= 0 && x < height && y >= 0 && y < width){
      sum += original[x * width + y] * f[i];
    }
  }
  return sum;
}
//Idea From lab10
//For here n is the itreation number, same as number of pixel at the beginning, the number of thread is 
//changed with n. If the block number is equal to one, then the kernel's normalization nearly done.
inline bool calculate_blocks_and_threads(int n, int &blocks, int &threads){
  if(n <= 512){threads = 512;}
  else if(n <= 256){threads = 256;}
  else if(n <= 128){threads = 128;}
  else if(n <= 64){threads = 64;}
  else if(n <= 32){threads = 32;}
  else if(n <= 16){threads = 16;}
  else if(n <= 8){threads = 8;}
  else if(n <= 4){threads = 4;}
  else if(n <= 2){threads = 2;}
  else{threads = 1024;}
  blocks = (n + (threads - 1)) / threads;
  return blocks != 1;
}

template <unsigned int blockSize>
__global__ void reduction_max(int32_t *biggest, int32_t *output, int n, int first){
  extern __shared__ int32_t sdata_max[];
  unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x) + threadIdx.x;
  if(first == 1){
    if(i < n){sdata_max[tid] = output[i];}
    else{sdata_max[tid] = INT_MIN;}
  }
  else{
    if(i < n){sdata_max[tid] = biggest[i];}
    else{sdata_max[tid] = INT_MIN;}
  }
  __syncthreads();

  // do reduction in shared memory
  if (blockSize >= 1024) { if (tid < 512){sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 512]);} __syncthreads();}
	if (blockSize >= 512) { if (tid < 256){sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 256]);} __syncthreads();}
	if (blockSize >= 256) { if (tid < 128){sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 128]);} __syncthreads();}
	if (blockSize >= 128) {	if (tid <  64){sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 64]);} __syncthreads();}
  
  if(tid < 32){
    if (blockSize >= 64) {sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 32]); __syncthreads();}
    if (blockSize >= 32) {sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 16]); __syncthreads();}
    if (blockSize >= 16) {sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 8]); __syncthreads();}
    if (blockSize >= 8) {sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 4]); __syncthreads();}
    if (blockSize >= 4) {sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 2]); __syncthreads();}
    if (blockSize >= 2) {sdata_max[tid] = MY_MAX(sdata_max[tid], sdata_max[tid + 1]); __syncthreads();}
  }
  //__syncthreads();
  //End, write result back to global memory
  if (blockSize >= 1) { if (tid == 0){biggest[blockIdx.x] = sdata_max[0];}}
}

template <unsigned int blockSize>
__global__ void reduction_min(int32_t *smallest, int32_t *output, int n, int first){
  extern __shared__ int32_t sdata_min[];
  unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x) + threadIdx.x;
  if(first == 1){
    if(i < n){sdata_min[tid] = output[i];}
    else{sdata_min[tid] = INT_MAX;}
  }
  else{
    if(i < n){sdata_min[tid] = smallest[i];}
    else{sdata_min[tid] = INT_MAX;}
  }
  __syncthreads();

  // do reduction in shared memory
  if (blockSize >= 1024) { if (tid < 512){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 512]);} __syncthreads();}
	if (blockSize >= 512) { if (tid < 256){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 256]);} __syncthreads();}
	if (blockSize >= 256) { if (tid < 128){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 128]);} __syncthreads();}
	if (blockSize >= 128) {	if (tid < 64){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 64]);} __syncthreads();}
  
  if(tid < 32){
    if (blockSize >= 64){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 32]); __syncthreads();}
    if (blockSize >= 32){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 16]); __syncthreads();}
    if (blockSize >= 16){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 8]); __syncthreads();}
    if (blockSize >= 8){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 4]); __syncthreads();}
    if (blockSize >= 4){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 2]); __syncthreads();}
    if (blockSize >= 2){sdata_min[tid] = MY_MIN(sdata_min[tid], sdata_min[tid + 1]); __syncthreads();}
  }
  //__syncthreads();
  //End, write result back to global memory
  if (blockSize >= 1) { if (tid == 0){smallest[blockIdx.x] = sdata_min[0];}}
}

inline void gpu_switch_threads(int nPixel, int num_threads, int num_blocks, int32_t *input, int32_t *smallest, int32_t *biggest, int first){
  int shMemSize = (num_threads <= 32) ? 2 * num_threads * sizeof(float) : num_threads * sizeof(float);
  switch (num_threads){
  case 1024:
    reduction_max<1024><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<1024><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 512:
    reduction_max<512><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<512><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 256:
    reduction_max<256><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<256><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 128:
    reduction_max<128><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<128><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 64:
    reduction_max<64><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<64><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 32:
    reduction_max<32><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<32><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 16:
    reduction_max<16><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<16><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 8:
    reduction_max<8><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<8><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 4:
    reduction_max<4><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<4><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 2:
    reduction_max<2><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<2><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  case 1:
    reduction_max<1><<<num_blocks, num_threads, shMemSize>>>(biggest, input, nPixel, first);
    reduction_min<1><<<num_blocks, num_threads, shMemSize>>>(smallest, input, nPixel, first);
    break;
  default:
    printf("invalid number of threads, exiting...\n");
    exit(1);
  }
}

float run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height);

void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *time_returned);
__global__ void kernel1(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *time_returned);
__global__ void kernel2(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize2(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *time_returned);
__global__ void kernel3(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t rows_per_thread);
__global__ void normalize3(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest, int32_t rows_per_thread);

void run_kernel4(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *time_returned);
__global__ void kernel4(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize4(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *time_returned);
/* This is your own kernel, you should decide which parameters to add
   here*/
__global__ void kernel5(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);
#endif
