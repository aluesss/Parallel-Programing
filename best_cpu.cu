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
#include <pthread.h>
#include <stdio.h>

void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,  
                     int32_t largest) {  
  if (smallest == largest) {  
    return;  
  }  
  
  target[pixel_idx] =  
      ((target[pixel_idx] - smallest) * 255) / (largest - smallest);  
}

typedef struct filter_t {  
  int32_t dimension;  
  const int8_t *matrix;  
} filter_struct; 

int32_t apply2d(const filter_struct *f, const int32_t *original, int32_t *target,  
                int32_t width, int32_t height, int row, int column) {  
  int dimension = f->dimension;  
  int total = dimension * dimension; //Total pixel in the filter  
  int center_row = dimension / 2, center_column = dimension / 2; //Find the center  
  int sum = 0;  
  for(int i=0; i<total; i++){  
    int current_row = i / dimension, current_column = i % dimension;  
    //Compute valid neighbour  
    int x = row + (current_row - center_row);  
    int y = column + (current_column - center_column);  
    if(x >= 0 && x < height && y >= 0 && y < width){  
      sum += original[x * width + y] * f->matrix[i];  
    }  
  }  
  return sum;  
}  

typedef struct common_work_t{  
    const filter_struct *f;  
    const int32_t *original_image;  
    int32_t *output_image;  
    int32_t width;  
    int32_t height;  
    int32_t *smallest;  
    int32_t *largest;  
    pthread_mutex_t *mutex;  
    int32_t rows;   
    int32_t cols;  
    int32_t first_assigned_row;  
    int32_t first_assigned_col;  
    int32_t max_threads;
} common_work;  

typedef struct work_t{  
    common_work *common;  
    int32_t id;  
} worker; 

//part3 structure  
typedef struct chunk_t{  
  int32_t row_start;  
  int32_t row_end;  
  int32_t column_start;  
  int32_t column_end;  
} chunk;

//Part three global vars  
int32_t chunk_amount;  
int32_t chunk_in_filter;  
int32_t chunk_in_normalization;  
chunk *queue;  

pthread_barrier_t sum_barrier;  
pthread_barrier_t normalize_barrier; 

void *queue_work(void *work) {  
  //initial work  
  worker* w = (worker*)work;  
  common_work* cm = w->common;  
  int t_smallest = INT_MAX, t_largest = INT_MIN;  
  //int32_t width = cm->width, height = cm->height;  
  //Apply filiters  
  while(chunk_in_filter < chunk_amount) {  
    chunk c;  
    pthread_mutex_lock(cm->mutex);  
    if(chunk_in_filter >= chunk_amount){  
      pthread_mutex_unlock(cm->mutex);  
      break;  
    }  
    c = queue[chunk_in_filter];  
    chunk_in_filter++;  
    pthread_mutex_unlock(cm->mutex);  
    int32_t sum;  
    for(int32_t i=c.row_start; i < c.row_end; i++){  
      for(int32_t j=c.column_start; j < c.column_end; j++){ // Get each element in height level   
        sum = apply2d(cm->f, cm->original_image, cm->output_image, cm->width, cm->height, i, j);  
        cm->output_image[i * cm->width + j] = sum;  
        if(sum < t_smallest){t_smallest = sum;}  
        if(sum > t_largest) {t_largest = sum;}  
      }  
    }  
  }  
  // Wait for all threads compute local max and min  
  pthread_barrier_wait(&sum_barrier);  
  // Compute global max and min  
  pthread_mutex_lock(w->common->mutex);  
  if(t_smallest < *w->common->smallest){*w->common->smallest = t_smallest;}  
  if(t_largest > *w->common->largest) {*w->common->largest = t_largest;}  
  pthread_mutex_unlock(w->common->mutex);  
  
  pthread_barrier_wait (&normalize_barrier);  
  //Normalization  
  while(chunk_in_normalization < chunk_amount){  
    chunk c;  
    pthread_mutex_lock(cm->mutex);  
    if(chunk_in_normalization >= chunk_amount){  
      pthread_mutex_unlock(cm->mutex);  
      break;  
    }  
    c = queue[chunk_in_normalization];  
    chunk_in_normalization++;  
    pthread_mutex_unlock(w->common->mutex);  
  
    for(int32_t i=c.row_start; i < c.row_end; i++){ // Progressively fill out each set of columns with the partial rows.   
      for(int32_t j=c.column_start; j < c.column_end; j++){   
        normalize_pixel(cm->output_image, i * cm->width + j, *cm->smallest, *cm->largest);  
      }  
    }  
  }  
  return NULL;   
}

float run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {
    // Best cpu was work pool
    int smallest = INT_MAX;  
    int largest = INT_MIN;
    pthread_mutex_t mutex;  
    pthread_mutex_init(&mutex, NULL);   
    int32_t jobs;  
    int32_t load;
    int32_t num_threads = 8;
    int32_t work_chunk = 4; 
    const filter_struct f = {dimension, filter};
    //filter_struct f = {dimension, filter};
    if(height < num_threads){ // height less then num threads -> one col per thread   
      jobs = height;  
      load = 1;   
    } else if (height % num_threads == 0 ){ // Each thead handles an even number of cols   
      jobs = num_threads;  
      load = height / num_threads;  
    } else if (height % num_threads != 0) {   
      jobs = num_threads;   
      load = (height+num_threads-1) / num_threads;  
    }  
    pthread_barrier_init (&sum_barrier, NULL, jobs);  
    pthread_barrier_init (&normalize_barrier, NULL, jobs);  
    pthread_t threads[jobs];  
    worker* working =(worker*) malloc(jobs *sizeof(worker));  
    //Initial chunk  
    chunk_amount = 0;  
    chunk_in_filter = 0;  
    chunk_in_normalization = 0;  
    queue = (chunk*) malloc(width * height * sizeof(chunk));  
    //Build the queue  
    int32_t total = width * height, used = 0;  
    int32_t row_start = 0, row_end = 0, column_start = 0, column_end = 0;  
    while(used < total){  
      int32_t curr_row = row_start + work_chunk, curr_col = column_start + work_chunk;  
      if(curr_row >= height){  
        row_end = height;  
      }else{  
        row_end = curr_row;  
      }  
      if(curr_col >= width){  
        column_end = width;  
      }else{  
        column_end = curr_col;  
      }  
      queue[chunk_amount].column_start = column_start;  
      queue[chunk_amount].row_start = row_start;  
      queue[chunk_amount].column_end = column_end;  
      queue[chunk_amount].row_end = row_end;  
      used += (column_end-column_start) * (row_end-row_start);  
      column_start = column_end;  
      if(column_start >= width){  
        column_start = 0;  
        row_start += work_chunk;  
      }  
      chunk_amount++;  
    }  
  
    int32_t remaining = height;   
    int32_t assigned = 0;   

    struct timespec start, stop;  
    clock_gettime(CLOCK_MONOTONIC, &start);  
    
    for (int i = 0; i < jobs; i++) {    
      working[i].id = i;  
      working[i].common = (common_work*) malloc(jobs *sizeof(common_work));  
      *working[i].common = {.f = &f, .original_image = input, .output_image = output, .width = width, .height = height, .smallest = &smallest, .largest = &largest , .mutex = &mutex, .rows = load, .cols = 0, .first_assigned_row = assigned, .first_assigned_col = 0, .max_threads = jobs};  
      assigned += load;   
      remaining -= load;  
      if (remaining < load ){  
        load = remaining;   
      }  
      pthread_create(&threads[i], NULL, queue_work, &working[i]);    
    }  
  
    for (int i = 0; i < jobs; i++) {    
      pthread_join(threads[i], NULL);    
    }
    clock_gettime(CLOCK_MONOTONIC, &stop); 

    float ms_time = ((stop.tv_sec * 1000000000) + stop.tv_nsec) -
           ((start.tv_sec * 1000000000) + start.tv_nsec);

    for (int i = 0; i < jobs; i++) {    
      free(working[i].common);  
    }   
    free(working);  
    free(queue);
    return ms_time / 1.0e6;
}