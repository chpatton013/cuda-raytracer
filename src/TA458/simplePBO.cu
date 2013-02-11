// simplePBO.cpp (Rob Farber)

// includes
#include <GL/glut.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "../Type.h"

extern camera_t camera;
extern sphere_t* spheres;
extern light_t* lights;
extern camera_t* d_camera;
extern sphere_t* d_spheres;
extern light_t* d_lights;
extern uint16_t sphere_count;
extern uint16_t light_count;

// external variables
extern float animTime;
extern unsigned int window_width;
extern unsigned int window_height;

// constants (the following should be a const in a header file)

extern "C" void launch_kernel(void* pos, unsigned int, unsigned int, float);

// variables
GLuint pbo=0;
GLuint textureID=0;

void createPBO(GLuint* pbo)
{

  if (pbo) {
    // set up vertex data parameter
    int num_texels = window_width * window_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}

void deletePBO(GLuint* pbo)
{
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);

    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);

    *pbo = 0;
  }
}

void createTexture(GLuint* textureID, unsigned int size_x, unsigned int size_y)
{
  // Enable Texturing
  glEnable(GL_TEXTURE_2D);

  // Generate a texture identifier
  glGenTextures(1,textureID);

  // Make this the current texture (remember that GL is state-based)
  glBindTexture( GL_TEXTURE_2D, *textureID);

  // Allocate the texture memory. The last parameter is NULL since we only
  // want to allocate memory, not initialize it
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0,
            GL_BGRA,GL_UNSIGNED_BYTE, NULL);

  // Must set the filter mode, GL_LINEAR enables interpolation when scaling
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
  // GL_TEXTURE_2D for improved performance if linear interpolation is
  // not desired. Replace GL_LINEAR with GL_NEAREST in the
  // glTexParameteri() call

}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);

    *tex = 0;
}

void cleanupCuda()
{
  if(pbo) deletePBO(&pbo);
  if(textureID) deleteTexture(&textureID);
   cudaFree(d_camera);
   cudaFree(d_spheres);
   cudaFree(d_lights);
}

// Run the Cuda part of the computation
void runCuda()
{
  uchar4 *dptr=NULL;

  // map OpenGL buffer object for writing from CUDA on a single GPU
  // no data is moved (Win & Linux). When mapped to CUDA, OpenGL
  // should not use this buffer
  cudaGLMapBufferObject((void**)&dptr, pbo);

  // execute the kernel
  launch_kernel(dptr, window_width, window_height, animTime);

  // unmap buffer object
  cudaGLUnmapBufferObject(pbo);
}

void initCuda()
{
  // First initialize OpenGL context, so we can properly set the GL
  // for CUDA.  NVIDIA notes this is necessary in order to achieve
  // optimal performance with OpenGL/CUDA interop.  use command-line
  // specified CUDA device, otherwise use device with highest Gflops/s
  int devCount= 0;
  cudaGetDeviceCount(&devCount);
  if( devCount < 1 )
  {
     printf("No GPUS detected\n");
     exit(EXIT_FAILURE);
  }
  cudaGLSetGLDevice( 0 );

  createPBO(&pbo);
  createTexture(&textureID,window_width,window_height);

   cudaMalloc((void**)&d_camera, sizeof(camera_t));
   cudaMalloc((void**)&d_spheres, sizeof(sphere_t) * sphere_count);
   cudaMalloc((void**)&d_lights, sizeof(light_t) * light_count);

   cudaMemcpy(d_camera, &camera, sizeof(camera_t), cudaMemcpyHostToDevice);
   cudaMemcpy(
      d_spheres, spheres, sizeof(sphere_t) * sphere_count,
      cudaMemcpyHostToDevice
   );
   cudaMemcpy(
      d_lights, lights, sizeof(light_t) * light_count,
      cudaMemcpyHostToDevice
   );

  // Clean up on program exit
  atexit(cleanupCuda);

  runCuda();
}

extern __host__ __device__ void cross(float* v1, float* v2, float* v_dest);
extern __host__ __device__ void copy(float* v_src, float* v_dest);
extern __host__ __device__ float dot(float* v1, float* v2);
extern __host__ __device__ void zero_vector(float* v);
extern __host__ __device__ void add(float* v1, float* v2, float* v_dest);
extern __host__ __device__ void add_i(float* v1, float* v2);
extern __host__ __device__ void sub(float* v1, float* v2, float* v_dest);
extern __host__ __device__ void sub_i(float* v1, float* v2);
extern __host__ __device__ void scale(float* v_src, float s, float* v_dest);
extern __host__ __device__ void scale_i(float* v, float s);
extern __host__ __device__ void norm(float* v_src, float* v_dest);
extern __host__ __device__ void norm_i(float* v);
extern __host__ __device__ void print(float* v);
extern float CoS[3]; // center of scene
const float rad_incr = M_PI / 100.0f;
void parallel(float* v, float* axis, float* res) {
   scale(v, dot(v, axis), res);
}
void perp(float* v, float* axis, float* res) {
   parallel(v, axis, res);
   sub(v, res, res);
}
void rotate(float* vec, float* point, float* axis, float radians, float* res) {
   float sin_theta = sin(radians),
         cos_theta = cos(radians);

   float a = point[0], b = point[1], c = point[2];
   float x = vec[0], y = vec[1], z = vec[2];
   float u = axis[0], v = axis[1], w = axis[2];

   float ux_vy_wz = u*x + v*y + w*z;

   res[0] = (a * (v*v + w*w) - u * (b*v + c*w - ux_vy_wz)) * (1 - cos_theta) +
      (x * cos_theta) + (-c*v + b*w - w*y + v*z) * sin_theta;
   res[1] = (b * (u*u + w*w) - v * (a*u + c*w - ux_vy_wz)) * (1 - cos_theta) +
      (y * cos_theta) + (c*u - a*w + w*x - u*z) * sin_theta;
   res[2] = (c * (u*u + v*v) - w * (a*u + b*v - ux_vy_wz)) * (1 - cos_theta) +
      (z * cos_theta) + (-b*u + a*v - v*x + u*y) * sin_theta;
}
void camera_move(float* axis, float radians) {
   float rotated_position[3];
   rotate(camera.position, CoS, axis, radians, rotated_position);

   // front
   float pos_min_cos[3];
   sub(CoS, rotated_position, pos_min_cos);
   norm(pos_min_cos, camera.front);

   // up
   float y_axis[3] = {0.0f, 1.0f, 0.0f};
   float right[3];
   float up[3];
   cross(camera.front, y_axis, right);
   cross(right, camera.front, up);
   norm(up, camera.up);

   // position
   copy(rotated_position, camera.position);

   cudaMemcpy(d_camera, &camera, sizeof(camera_t), cudaMemcpyHostToDevice);
}
void camera_move_up_down(float radians) {
   float axis[3];
   cross(camera.front, camera.up, axis);
   norm_i(axis);

   camera_move(axis, radians);
}
void camera_move_up() {
   camera_move_up_down(rad_incr);
}
void camera_move_down() {
   camera_move_up_down(-rad_incr);
}
void camera_move_left_right(float radians) {
   float axis[3];
   copy(camera.up, axis);
   norm_i(axis);

   camera_move(axis, radians);
}
void camera_move_left() {
   camera_move_left_right(-rad_incr);
}
void camera_move_right() {
   camera_move_left_right(rad_incr);
}
