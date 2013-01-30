#include <stdio.h>


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

__global__ void MMKernel(int i)
{
  printf("world\n");
  
  printf("world\n");
}

void cudaWrapper()
{

  int c;
  int *dev_c;
  
  HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );
  
  MMKernel<<<1,1>>>(1);
  
  
  HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost ) );

  HANDLE_ERROR( cudaFree( dev_c ) );  
  
  printf("hello\n");  

}