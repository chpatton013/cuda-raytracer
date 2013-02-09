//kernelPBO.cu (Rob Farber)

#include <stdio.h>
#include <stdint.h>
#include "../Type.h"

extern __global__ void draw_scene_kernel(
   sphere_t* d_spheres, uint16_t sphere_count,
   light_t* d_lights, uint16_t light_count,
   camera_t* d_camera, uchar4* pos, uint16_t win_w, uint16_t win_h
);
extern camera_t* d_camera;
extern sphere_t* d_spheres;
extern light_t* d_lights;
extern uint16_t sphere_count;
extern uint16_t light_count;

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(uchar4* pos, unsigned int window_width,
                  unsigned int window_height, float time)
{
   int block_width = 16;
   dim3 threads = dim3(block_width, block_width);
   dim3 blocks = dim3(
      window_width / block_width + ((window_width % block_width) ? 1 : 0),
      window_height / block_width + ((window_height % block_width) ? 1 : 0)
   );

   draw_scene_kernel<<<blocks,threads>>>(
      d_spheres, sphere_count, d_lights, light_count,
      d_camera, pos, window_width, window_height
   );

  // make certain the kernel has completed
  cudaThreadSynchronize();

  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s.\n", cudaGetErrorString( err) );
    exit(EXIT_FAILURE);
  }
}
