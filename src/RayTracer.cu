#include "RayTracer.h"

#define CUDA_CALLABLE __host__ __device__

bool initialize_cuda(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, uint16_t win_w, uint16_t win_h
) {
   int devCount = 0;
   cudaGetDeviceCount(&devCount);
   if (devCount < 1) {
      printf("No CUDA devices detected\n");
      exit(EXIT_FAILURE);
   }
   cudaGLSetGLDevice(0);

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

   create_pbo(&pbo_handle, win_w, win_h);
   create_texture(&texture_handle, win_w, win_h);

   d_sphere_count = sphere_count;
   d_light_count = light_count;

   return true;
}

void cleanup_cuda() {
   cudaFree(d_camera);
   cudaFree(d_spheres);
   cudaFree(d_lights);

   destroy_pbo(&pbo_handle);
   destroy_texture(&texture_handle);
}

void create_pbo(GLuint* pbo, uint16_t win_w, uint16_t win_h) {
   int num_texels = win_w * win_h;
   int num_values = num_texels * 4;
   int size_tex_data = sizeof(GLubyte) * num_values;

   glGenBuffers(1, pbo);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);

   glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);

   cudaGLRegisterBufferObject(*pbo);
}

void destroy_pbo(GLuint* pbo) {
   cudaGLUnregisterBufferObject(*pbo);

   glBindBuffer(GL_ARRAY_BUFFER, *pbo);
   glDeleteBuffers(1, pbo);
}

void create_texture(GLuint* tex, uint16_t win_w, uint16_t win_h) {
   glGenTextures(1, tex);
   glBindTexture(GL_TEXTURE_2D, *tex);

   glTexImage2D(
      GL_TEXTURE_2D, 0, GL_RGBA8,
      win_w, win_h, 0,
      GL_BGRA,GL_UNSIGNED_BYTE, NULL
   );

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
   // GL_TEXTURE_2D for improved performance if linear interpolation is
   // not desired. Replace GL_LINEAR with GL_NEAREST in the
   // glTexParameteri() call
}

void destroy_texture(GLuint* tex) {
   glDeleteTextures(1, tex);
}

void draw_scene() {
   uchar4* d_pixel_buffer = NULL;
   cudaGLMapBufferObject((void**)&d_pixel_buffer, pbo_handle);

   int block_width = 16;
   dim3 threads = dim3(block_width, block_width);
   dim3 blocks = dim3(
      win_w / block_width + ((win_w % block_width) ? 1 : 0),
      win_h / block_width + ((win_h % block_width) ? 1 : 0)
   );

   draw_scene_kernel<<<blocks,threads>>>(
      d_spheres, d_sphere_count, d_lights, d_light_count,
      d_camera, d_pixel_buffer, win_w, win_h
   );
   cudaThreadSynchronize();

   cudaGLUnmapBufferObject(pbo_handle);
}

__global__ void draw_scene_kernel(
   sphere_t* d_spheres, uint16_t sphere_count,
   light_t* d_lights, uint16_t light_count,
   camera_t* d_camera, uchar4* d_pixel_buffer,
   uint16_t win_w, uint16_t win_h
) {
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   if (x < win_w && y < win_h) {
      cast_primary_ray(
         d_lights, light_count, d_spheres, sphere_count,
         d_camera, d_pixel_buffer, win_w, win_h, x, y
      );
   }
}

CUDA_CALLABLE void cast_primary_ray(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, uchar4* pixel_buffer,
   uint16_t win_w, uint16_t win_h,
   uint16_t x, uint16_t y
) {
   ray_t ray;
   copy(camera->position, ray.origin);
   get_primary_ray_direction(camera, win_w, win_h, x, y, ray.direction);

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

   int index = y * win_w + x;
   pixel_buffer[index].x = (unsigned char)(color[0] * 255);
   pixel_buffer[index].y = (unsigned char)(color[1] * 255);
   pixel_buffer[index].z = (unsigned char)(color[2] * 255);
   pixel_buffer[index].w = 255;
}

CUDA_CALLABLE void get_primary_ray_direction(
   camera_t* camera,
   uint16_t win_w, uint16_t win_h,
   uint16_t x, uint16_t y,
   float* direction
) {
   float scaled_camera_right[3];
   cross(camera->front, camera->up, scaled_camera_right);
   scale_i(
      scaled_camera_right,
      tan(camera->fov[0] * 0.5f) *
       (((x + 0.5f) * 2.0f / (float)win_w) - 1.0f)
   );

   float scaled_camera_up[3];
   scale(
      camera->up,
      tan(camera->fov[1] * 0.5f) *
       (1.0f - ((win_h - y + 0.5f) / (float)win_h) * 2.0f),
      scaled_camera_up
   );

   // ray_direction = normalized(
   //    camera forward + scaled_camera_right + scaled_camera_up
   // )
   add(camera->front, scaled_camera_right, direction);
   add_i(direction, scaled_camera_up);
   norm_i(direction);
}

CUDA_CALLABLE float get_closest_intersection(
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

CUDA_CALLABLE void light_surface(
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

#undef CUDA_CALLABLE
