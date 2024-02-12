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
#include <string>
#include <unistd.h>
#include "kernels.h"
#include "pgm.h"

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
               float time_gpu_transfer_in, float time_gpu_transfer_out) {
  printf("%12.6f ", time_cpu);
  printf("%5d ", kernel);
  printf("%12.6f ", time_gpu_computation);
  printf("%14.6f ", time_gpu_transfer_in);
  printf("%15.6f ", time_gpu_transfer_out);
  printf("%13.2f ", time_cpu / time_gpu_computation);
  printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                                time_gpu_transfer_out));
}

int main(int argc, char **argv) {
  int c;
  std::string input_filename, cpu_output_filename, base_gpu_output_filename;
  if (argc < 3) {
    printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
    return 0;
  }

  while ((c = getopt(argc, argv, "i:o:")) != -1) {
    switch (c) {
    case 'i':
      input_filename = std::string(optarg);
      break;
    case 'o':
      cpu_output_filename = std::string(optarg);
      base_gpu_output_filename = std::string(optarg);
      break;
    default:
      return 0;
    }
  }

  pgm_image source_img;
  init_pgm_image(&source_img);

  if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR) {
    printf("Error loading source image.\n");
    return 0;
  }

  /* Do not modify this printf */
  printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
         "Speedup_noTrf Speedup\n");

  float time_cpu = 0.0;

  /* TODO: run your CPU implementation here and get its time. Don't include
   * file IO in your measurement. Store the time taken in a variable, so
   * it can be printed later for comparison with GPU kernels. */
  /* For example: */
  const int8_t filter_matrix[] = {
   0, 1, 0, 1, -4, 1, 0, 1, 0,
  };
  const int filter_dim = 3;
  
  {
    std::string cpu_file = cpu_output_filename;
    pgm_image cpu_output_img;
    copy_pgm_image_size(&source_img, &cpu_output_img);
    // Start time
    // time_cpu = run_best_cpu(args...);  // From kernels.h
    time_cpu = run_best_cpu(filter_matrix, filter_dim, source_img.matrix,
                  cpu_output_img.matrix, source_img.width, source_img.height);
    // End time
    save_pgm_to_file(cpu_file.c_str(), &cpu_output_img);
    destroy_pgm_image(&cpu_output_img);
  }

  /* TODO:
   * run each of your gpu implementations here,
   * get their time,
   * and save the output image to a file.
   * Don't forget to add the number of the kernel
   * as a prefix to the output filename:
   * Print the execution times by calling print_run().
   */

  /* For example: */
  //Kernal 1
  
  float *time_returned =(float *) malloc(sizeof(float)*3); 
  /**/
  {
    std::string gpu_file = "1" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    // Start time
    // run_kernel1(args...);  // From kernels.h
    run_kernel1(filter_matrix, filter_dim, source_img.matrix,
                  gpu_output_img.matrix, source_img.width, source_img.height, time_returned);
    

    // End time
    // print_run(args...)     // Defined on the top of this file
    print_run(time_cpu, 1, time_returned[0], time_returned[1], time_returned[2]);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
    free(time_returned); 
  }
  /**/
 
  //Kernal 2
  time_returned =(float *) malloc(sizeof(float)*3); 
  {
    std::string gpu_file = "2" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    // Start time
    // run_kernel1(args...);  // From kernels.h
    run_kernel2(filter_matrix, filter_dim, source_img.matrix,
                  gpu_output_img.matrix, source_img.width, source_img.height, time_returned);

    // End time
    // print_run(args...)     // Defined on the top of this file
    print_run(time_cpu, 2, time_returned[0], time_returned[1], time_returned[2]);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
    free(time_returned); 
  }
  /**/
  //Kernal 3
  time_returned =(float *) malloc(sizeof(float)*3); 
  {
    std::string gpu_file = "3" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    // Start time
    // run_kernel1(args...);  // From kernels.h
    run_kernel3(filter_matrix, filter_dim, source_img.matrix,
                  gpu_output_img.matrix, source_img.width, source_img.height, time_returned);

    // End time
    // print_run(args...)     // Defined on the top of this file
    print_run(time_cpu, 3, time_returned[0], time_returned[1], time_returned[2]);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
    free(time_returned); 
  }
  /**/
  /**/
  //Kernal 4
  time_returned =(float *) malloc(sizeof(float)*3); 
  {
    std::string gpu_file = "4" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    // Start time
    // run_kernel1(args...);  // From kernels.h
    run_kernel4(filter_matrix, filter_dim, source_img.matrix,
                  gpu_output_img.matrix, source_img.width, source_img.height, time_returned);

    // End time
    // print_run(args...)     // Defined on the top of this file
    print_run(time_cpu, 4, time_returned[0], time_returned[1], time_returned[2]);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
    free(time_returned);
  }
  /**/
   
  //Kernal 5
  time_returned =(float *) malloc(sizeof(float)*3); 
  {
    std::string gpu_file = "5" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    // Start time
    // run_kernel1(args...);  // From kernels.h
    run_kernel5(filter_matrix, filter_dim, source_img.matrix,
                  gpu_output_img.matrix, source_img.width, source_img.height, time_returned);

    // End time
    // print_run(args...)     // Defined on the top of this file
    print_run(time_cpu, 5, time_returned[0], time_returned[1], time_returned[2]);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
    free(time_returned);
  }
   
}
