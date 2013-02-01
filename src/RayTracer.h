#ifndef _RAY_TRACER_H_
#define _RAY_TRACER_H_

#include <stdint.h>
#include <cuda.h>
#include "Type.h"
#include "Util.h"

void draw_scene(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, float* img_buffer,
   uint16_t img_w, uint16_t img_h,
   bool cpu_mode
);
void draw_scene_cpu(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, float* img_buffer,
   uint16_t img_w, uint16_t img_h
);
void draw_scene_gpu(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, float* img_buffer,
   uint16_t img_w, uint16_t img_h
);
__global__ void draw_scene_kernel(
   Sphere* d_spheres, uint16_t sphere_count,
   Light* d_lights, uint16_t light_count,
   Camera* d_camera, float* d_image_buffer,
   uint16_t img_w, uint16_t img_h
);

CUDA_CALLABLE void cast_primary_ray(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, float* img_buffer,
   uint16_t img_w, uint16_t img_h,
   uint16_t x, uint16_t y
);
CUDA_CALLABLE void get_primary_ray_direction(
   camera_t* camera,
   uint16_t img_w, uint16_t img_h,
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

#endif
