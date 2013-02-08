#ifndef _RAY_TRACER_H_
#define _RAY_TRACER_H_

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include "Timer.h"
#include "Type.h"
#include "Util.h"

static uint16_t win_w, win_h;
static uint16_t sphere_c, light_c;
static GLuint pbo = 0;
static GLuint textureID = 0;
static camera_t* d_camera;
static sphere_t* d_spheres;
static light_t* d_lights;
static float* d_img_buffer;
static int block_width = 16;
static dim3 thread_dim;
static dim3 block_dim;
static Timer timer;

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

void initialize_cuda_context(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, uint16_t img_w, uint16_t img_h
);
void draw_scene();
void destroy_cuda_context();
void createPBO(GLuint* pbo, uint16_t img_w, uint16_t img_h);
void deletePBO(GLuint* pbo);
void createTexture(GLuint* tex, unsigned int img_w, unsigned int img_h);
void deleteTexture(GLuint* tex);

__host__ __device__ void get_primary_ray_direction(
   camera_t* camera,
   uint16_t img_w, uint16_t img_h,
   uint16_t x, uint16_t y,
   float* direction
) {
   float scaled_camera_right[3];
   cross(camera->front, camera->up, scaled_camera_right);
   scale_i(
      scaled_camera_right,
      tan(camera->fov[0] * 0.5f) *
       (((x + 0.5f) * 2.0f / (float)img_w) - 1.0f)
   );

   float scaled_camera_up[3];
   scale(
      camera->up,
      tan(camera->fov[1] * 0.5f) *
       (1.0f - ((img_h - y + 0.5f) / (float)img_h) * 2.0f),
      scaled_camera_up
   );

   // ray_direction = normalized(
   //    camera forward + scaled_camera_right + scaled_camera_up
   // )
   add(camera->front, scaled_camera_right, direction);
   add_i(direction, scaled_camera_up);
   norm_i(direction);
}

__host__ __device__ float get_closest_intersection(
   ray_t* ray, sphere_t* spheres, uint16_t sphere_count, sphere_t** target
) {
   float closest_param = FLT_MAX;
   sphere_t* closest_sphere = NULL;

   for (int ndx = 0; ndx < sphere_count; ++ndx) {
      float current_param = get_ray_sphere_intersection(ray, spheres + ndx);

      if (current_param >= 0.0f && current_param < closest_param) {
         closest_param = current_param;
         closest_sphere = spheres + ndx;
      }
   }

   if (closest_sphere == NULL) {
      return -1.0f;
   } else {
      *target = closest_sphere;
      return closest_param;
   }
}

__host__ __device__ void light_surface(
   ray_t* ray, float parameter, sphere_t* sphere,
   camera_t* camera, light_t* lights, uint16_t light_count,
   float* color
) {
   float position[3];
   float surface_normal[3];
   float viewer_direction[3];

   // position = ray position(parameter)
   get_ray_position(ray, parameter, position);
   // normal = sphere normal(position)
   get_sphere_normal(sphere, position, surface_normal);
   // viewer = normalized(position - camera position)
   sub(position, camera->position, viewer_direction);
   norm_i(viewer_direction);

   for (int ndx = 0; ndx < light_count; ++ndx) {
      float light_direction[3];
      float reflected_direction[3];

      // light_direction = normalized(position - light position)
      sub(position, lights[ndx].position, light_direction);
      norm_i(light_direction);

      // reflected = normalized(
      //    (2 * (light_direction dot normal) * normal) - light_direction
      // )
      scale(
         surface_normal,
         (2 * dot(light_direction, surface_normal)),
         reflected_direction
      );
      sub_i(reflected_direction, light_direction);
      norm_i(reflected_direction);

      float lighting[3];
      get_phong_lighting(
         light_direction, surface_normal,
         reflected_direction, viewer_direction,
         &sphere->composition, lighting
      );
      add_i(color, lighting);
   }
}

__host__ __device__ void cast_primary_ray(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, float* img_buffer,
   uint16_t img_w, uint16_t img_h,
   uint16_t x, uint16_t y
) {
   ray_t ray;
   copy(camera->position, ray.origin);
   get_primary_ray_direction(camera, img_w, img_h, x, y, ray.direction);

   sphere_t* sphere;
   float intersection = get_closest_intersection(
      &ray, spheres, sphere_count, &sphere
   );

   float color[3] = {0.0f, 0.0f, 0.0f};
   if (intersection >= 0.0f) {
      light_surface(
         &ray, intersection, sphere,
         camera, lights, light_count, color
      );
   }
   copy(color, img_buffer + 3 * (y * img_w + x));
}

__global__ void draw_scene_kernel(
   sphere_t* d_spheres, uint16_t sphere_count,
   light_t* d_lights, uint16_t light_count,
   camera_t* d_camera, float* d_img_buffer,
   uint16_t img_w, uint16_t img_h
) {
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   if (x < img_w && y < img_h) {
      cast_primary_ray(
         d_lights, light_count, d_spheres, sphere_count,
         d_camera, d_img_buffer, img_w, img_h, x, y
      );
   }
}

void initialize_cuda_context(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, uint16_t img_w, uint16_t img_h
) {
   int devCount = 0;
   cudaGetDeviceCount(&devCount);
   if (devCount < 1) {
      printf("No CUDA devices detected\n");
      exit(EXIT_FAILURE);
   }
   cudaGLSetGLDevice(0);

   createPBO(&pbo, img_w, img_h);
   createTexture(&textureID, img_w, img_h);

   cudaMalloc((void**)&d_camera, sizeof(camera_t));
   cudaMalloc((void**)&d_spheres, sizeof(sphere_t) * sphere_count);
   cudaMalloc((void**)&d_lights, sizeof(light_t) * light_count);

   cudaMemcpy(d_camera, camera, sizeof(camera_t), cudaMemcpyHostToDevice);
   cudaMemcpy(
      d_spheres, spheres, sizeof(sphere_t) * sphere_count,
      cudaMemcpyHostToDevice
   );
   cudaMemcpy(
      d_lights, lights, sizeof(light_t) * light_count,
      cudaMemcpyHostToDevice
   );

   thread_dim = dim3(block_width, block_width);
   block_dim = dim3(
      img_w / block_width + ((img_w % block_width) ? 1 : 0),
      img_h / block_width + ((img_h % block_width) ? 1 : 0)
   );

   win_w = img_w;
   win_h = img_h;

   sphere_c = sphere_count;
   light_c = light_count;
}

void draw_scene() {
   float *dptr = NULL;
   cudaGLMapBufferObject((void**)&dptr, pbo);

   draw_scene_kernel<<<block_dim,thread_dim>>>(
      d_spheres, sphere_c, d_lights, light_c,
      d_camera, dptr, win_w, win_h
   );
   cudaThreadSynchronize();

   cudaGLUnmapBufferObject(pbo);
}

void destroy_cuda_context() {
   cudaFree(d_camera);
   cudaFree(d_spheres);
   cudaFree(d_lights);
   /* cudaFree(d_img_buffer); */

  if (pbo) deletePBO(&pbo);
  if (textureID) deleteTexture(&textureID);
}

void createPBO(GLuint* pbo, uint16_t img_w, uint16_t img_h) {
   if (pbo) {
      int num_texels = img_w * img_h;
      int num_values = num_texels * 3;
      int size_tex_data = sizeof(GLubyte) * num_values;

      glGenBuffers(1, pbo);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
      cudaGLRegisterBufferObject(*pbo);
   }
}

void deletePBO(GLuint* pbo) {
   if (pbo) {
      cudaGLUnregisterBufferObject(*pbo);
      glBindBuffer(GL_ARRAY_BUFFER, *pbo);
      glDeleteBuffers(1, pbo);
      *pbo = 0;
   }
}

void createTexture(GLuint* tex, unsigned int img_w, unsigned int img_h) {
   if (tex) {
      glEnable(GL_TEXTURE_2D);
      glGenTextures(1, tex);
      glBindTexture(GL_TEXTURE_2D, *tex);

      glTexImage2D(
         GL_TEXTURE_2D, 0, GL_RGBA8,
         img_w, img_h, 0,
         GL_BGRA, GL_UNSIGNED_BYTE, NULL
      );

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
      // GL_TEXTURE_2D for improved performance if linear interpolation is
      // not desired. Replace GL_LINEAR with GL_NEAREST in the
      // glTexParameteri() call
   }
}

void deleteTexture(GLuint* tex) {
   if (tex) {
      glDeleteTextures(1, tex);
      *tex = 0;
   }
}

void display_fps(Timer* timer) {
   static int call_counter = 0;
   static int call_threshold = 100;
   static char str[256];

   if (call_counter == call_threshold) {
      float ms = timer->get() / 1000000.0f;
      sprintf(str, "%3.4f FPS - %3.4f MS", 1000.0f / ms, ms);
      glutSetWindowTitle(str);
      call_counter = 0;
   }

   ++call_counter;
}

void display() {
   timer.start();

   draw_scene();

   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
   glBindTexture(GL_TEXTURE_2D, textureID);

   glTexSubImage2D(
      GL_TEXTURE_2D, 0, 0,
      0, win_w, win_h,
      GL_RGBA, GL_UNSIGNED_BYTE, NULL
   );

   glBegin(GL_QUADS);
   glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,0.0f,0.0f);
   glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,0.0f,0.0f);
   glEnd();

   glutSwapBuffers();

   timer.stop();

   display_fps(&timer);
}

void keyboard(unsigned char key, int x, int y) {
   glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
   glutPostRedisplay();
}

void motion(int x, int y) {
   glutPostRedisplay();
}

#endif
