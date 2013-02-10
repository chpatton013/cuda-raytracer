#ifndef _RAY_TRACER_H_
#define _RAY_TRACER_H_

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "Type.h"
#include "Util.h"

void draw_scene(
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
   } else if (target != NULL) {
      *target = closest_sphere;
   }

   return closest_param;
}

__host__ __device__ void light_surface(
   ray_t* ray, float parameter, sphere_t* sphere,
   camera_t* camera, light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   float* color
) {
   const float EPSILON = 0.005f;

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

      // light_direction = normalized(position - light position)
      sub(position, lights[ndx].position, light_direction);
      norm_i(light_direction);

      ray_t light_ray;
      scale(light_direction, -1.0f, light_ray.direction);

      float pos_delta[3];
      scale(light_ray.direction, EPSILON, pos_delta);
      add(position, pos_delta, light_ray.origin);

      float light_dist = get_ray_light_intersection(&light_ray, &lights[ndx]);
      float closest_sphere = get_closest_intersection(
         &light_ray, spheres, sphere_count, NULL
      );

      bool shadowed = closest_sphere > EPSILON && closest_sphere < light_dist;
      if (!shadowed) {
         // reflected = normalized(
         //    (2 * (light_direction dot normal) * normal) - light_direction
         // )
         float reflected_direction[3];
         scale(
            surface_normal,
            (2 * dot(light_direction, surface_normal)),
            reflected_direction
         );
         sub_i(reflected_direction, light_direction);
         norm_i(reflected_direction);

         float diffuse[3],
               specular[3];
         get_diffuse_lighting(
            light_direction, surface_normal,
            &sphere->composition, diffuse
         );
         get_specular_lighting(
            reflected_direction, viewer_direction,
            &sphere->composition, specular
         );

         add_i(color, diffuse);
         add_i(color, specular);
      }
   }

   float ambient[3];
   get_ambient_lighting(&sphere->composition, ambient);
   add_i(color, ambient);
}

__host__ __device__ void cast_primary_ray(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, uchar4* img_buffer,
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
         camera, lights, light_count,
         spheres, sphere_count,
         color
      );
   }

   clamp(color, 0.0f, 1.0f);
   int index = y * img_w + x;
   img_buffer[index].x = (unsigned char)(color[0] * 255);
   img_buffer[index].y = (unsigned char)(color[1] * 255);
   img_buffer[index].z = (unsigned char)(color[2] * 255);
   img_buffer[index].w = 255;
}

__global__ void draw_scene_kernel(
   sphere_t* d_spheres, uint16_t sphere_count,
   light_t* d_lights, uint16_t light_count,
   camera_t* d_camera, uchar4* d_img_buffer,
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

/*
void draw_scene(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, float* img_buffer,
   uint16_t img_w, uint16_t img_h
) {
   camera_t* d_camera;
   sphere_t* d_spheres;
   light_t* d_lights;

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

   int block_width = 16;
   dim3 threads = dim3(block_width, block_width);
   dim3 blocks = dim3(
      img_w / block_width + ((img_w % block_width) ? 1 : 0),
      img_h / block_width + ((img_h % block_width) ? 1 : 0)
   );

   draw_scene_kernel<<<blocks,threads>>>(
      d_spheres, sphere_count, d_lights, light_count,
      d_camera, d_img_buffer, img_w, img_h
   );

   cudaFree(d_camera);
   cudaFree(d_spheres);
   cudaFree(d_lights);
}
*/

#endif
