#include "cuda.h"
#include <stdio.h>
#include <math.h>

static void HandleError( cudaError_t err,
  const char *file,
  int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
        file, line );
    exit( EXIT_FAILURE );
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


/*
 * Cuda Helper Functions (Ported from Util.cc)
 */
 
__device__ void cudaCopy(float* v_src, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v_src[ndx];
   }
}
__device__ float cudaDot(float* v1, float* v2) {
   float result = 0.0f;
   for (int ndx = 0; ndx < 3; ++ndx) {
      result += v1[ndx] * v2[ndx];
   }
   return result;
}
__device__ float cudaMag(float* v) {
   return sqrt(cudaDot(v, v));
}
__device__ void cudaCross(float* v1, float* v2, float* v_dest) {
   v_dest[0] = v1[1] * v2[2] - v1[2] * v2[1];
   v_dest[1] = v1[2] * v2[0] - v1[0] * v2[2];
   v_dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
}
__device__ void cudaScalei(float* v, float s) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v[ndx] *= s;
   }
}
__device__ void cudaScaleiArray(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] *= v2[ndx];
   }
}
__device__ void cudaScale(float* v, float s) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v[ndx] *= s;
   }
}
__device__ void cudaScaleDest(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] * v2[ndx];
   }
}
__device__ void cudaNorm(float* v_src, float* v_dest) {
   cudaScaleDest(v_src, 1.0f / cudaMag(v_src), v_dest); // error here. float pointer issue
}
__device__ void cudaNormi(float* v) {
   cudaScalei(v, 1.0f / cudaMag(v));
}
__device__ void cudaAdd(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] + v2[ndx];
   }
}
__device__ void cudaAddi(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] += v2[ndx];
   }
}




/*
 * Main Cuda Kernel
 */
__global__ void MMKernel(int i)
{
  printf("world\n");
  
}


/*
 * Cuda Wrapper called from Draw() function
 */
void cudaWrapper(int image_width, int image_height)
{

  int c;
  int *dev_c;
 
  HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );
  
  MMKernel<<<1,1>>>(1);
  
  
  HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost ) );

  HANDLE_ERROR( cudaFree( dev_c ) );  
  
  printf("hello\n");  

}