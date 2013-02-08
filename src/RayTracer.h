#ifndef _RAY_TRACER_H_
#define _RAY_TRACER_H_

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "Timer.h"
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

void draw_scene_cpu(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, float* img_buffer,
   uint16_t img_w, uint16_t img_h
) {
   uint64_t rays_cast = 0;

   for (uint16_t x = 0; x < img_w; ++x) {
      for (uint16_t y = 0; y < img_h; ++y) {
         cast_primary_ray(
            lights, light_count, spheres, sphere_count,
            camera, img_buffer, img_w, img_h, x, y
         );
         ++rays_cast;
      }

      if (x % (img_w / 25) == 0) {
         float complete = (x / (float)img_w);
         printf("%d%% complete...\n", (int)(complete * 100.0f));
      }
   }

   printf("DONE: %llu rays cast\n", (long long unsigned int)rays_cast);
}

void draw_scene_gpu(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, float* img_buffer,
   uint16_t img_w, uint16_t img_h
) {
   camera_t* d_camera;
   sphere_t* d_spheres;
   light_t* d_lights;
   float* d_img_buffer;

   cudaMalloc((void**)&d_camera, sizeof(camera_t));
   cudaMalloc((void**)&d_spheres, sizeof(sphere_t) * sphere_count);
   cudaMalloc((void**)&d_lights, sizeof(light_t) * light_count);
   cudaMalloc((void**)&d_img_buffer, sizeof(float) * img_w * img_h * 3);

   cudaMemcpy(d_camera, camera, sizeof(camera_t), cudaMemcpyHostToDevice);
   cudaMemcpy(
      d_spheres, spheres, sizeof(sphere_t) * sphere_count,
      cudaMemcpyHostToDevice
   );
   cudaMemcpy(
      d_lights, lights, sizeof(light_t) * light_count,
      cudaMemcpyHostToDevice
   );
   cudaMemcpy(
      d_img_buffer, img_buffer, sizeof(float) * img_w * img_h * 3,
      cudaMemcpyHostToDevice
   );

   int block_width = 16;
   dim3 threads = dim3(block_width, block_width);
   dim3 blocks = dim3(
      img_w / block_width + ((img_w % block_width) ? 1 : 0),
      img_h / block_width + ((img_h % block_width) ? 1 : 0)
   );

   static Timer timer;
   timer.start();

   draw_scene_kernel<<<blocks,threads>>>(
      d_spheres, sphere_count, d_lights, light_count,
      d_camera, d_img_buffer, img_w, img_h
   );
   cudaThreadSynchronize();

   timer.stop();
   printf("Computation complete: %lf ms\n", timer.get() / 1000000.0);

   cudaMemcpy(
      img_buffer, d_img_buffer, sizeof(float) * img_w * img_h * 3,
      cudaMemcpyDeviceToHost
   );

   cudaFree(d_camera);
   cudaFree(d_spheres);
   cudaFree(d_lights);
   cudaFree(d_img_buffer);
}

void draw_scene(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, float* img_buffer,
   uint16_t img_w, uint16_t img_h,
   bool cpu_mode
) {
   camera->fov[0] *= M_PI / 180.0f; // degrees to radians
   camera->fov[1] = camera->fov[0] * img_h / (float)img_w;

   if (cpu_mode) {
      draw_scene_cpu(
         lights, light_count, spheres, sphere_count,
         camera, img_buffer, img_w, img_h
      );
   } else {
      draw_scene_gpu(
         lights, light_count, spheres, sphere_count,
         camera, img_buffer, img_w, img_h
      );
   }
}

#endif
