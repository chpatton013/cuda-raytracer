#ifndef _RAY_TRACER_H_
#define _RAY_TRACER_H_

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include "Timer.h"
#include "Type.h"
#include "Util.h"

#define CUDA_CALLABLE __host__ __device__

// Externs
extern uint16_t win_w;
extern uint16_t win_h;

// Globals
static camera_t* d_camera;
static sphere_t* d_spheres;
static uint16_t d_sphere_count;
static light_t* d_lights;
static uint16_t d_light_count;
GLuint pbo_handle;
GLuint texture_handle;

bool initialize_cuda(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, uint16_t win_w, uint16_t win_h
);
void cleanup_cuda();

void create_pbo(GLuint* pbo, uint16_t win_w, uint16_t win_h);
void destroy_pbo(GLuint* pbo);
void create_texture(GLuint* tex, uint16_t win_w, uint16_t win_h);
void destroy_texture(GLuint* tex);

void draw_scene();

__global__ void draw_scene_kernel(
   sphere_t* d_spheres, uint16_t sphere_count,
   light_t* d_lights, uint16_t light_count,
   camera_t* d_camera, uchar4* d_pixel_buffer,
   uint16_t win_w, uint16_t win_h
);

CUDA_CALLABLE void cast_primary_ray(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, uchar4* pixel_buffer,
   uint16_t win_w, uint16_t win_h,
   uint16_t x, uint16_t y
);
CUDA_CALLABLE void get_primary_ray_direction(
   camera_t* camera,
   uint16_t win_w, uint16_t win_h,
   uint16_t x, uint16_t y,
   float* direction
);
CUDA_CALLABLE float get_closest_intersection(
   ray_t* ray, sphere_t* spheres, uint16_t sphere_count, sphere_t** target
);
CUDA_CALLABLE void light_surface(
   ray_t* ray, float parameter, sphere_t* sphere,
   camera_t* camera, light_t* lights, uint16_t light_count,
   float* color
);

#undef CUDA_CALLABLE

#endif
