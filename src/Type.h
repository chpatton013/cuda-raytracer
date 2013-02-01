#ifndef _TYPE_H_
#define _TYPE_H_

#include "Util.h"

struct light_t {
   float position[3];
   float color[3];
};

struct composition_t {
   float ambient[3];
   float diffuse[3];
   float specular[3];
   float shine;
};

struct sphere_t {
   float center[3];
   float radius;
   composition_t composition;
};

struct ray_t {
   float origin[3];
   float direction[3];
};

struct camera_t {
   float position[3];
   float front[3];
   float up[3];
   float z[3]; // near, far, focal
   float fov[2]; // horiz, vert: degrees
};

CUDA_CALLABLE void get_ambient_lighting(
   composition_t* composition, float* color
);
CUDA_CALLABLE void get_diffuse_lighting(
   float* light_direction, float* surface_normal,
   composition_t* composition, float* color
);
CUDA_CALLABLE void get_specular_lighting(
   float* reflected_direction, float* viewer_direction,
   composition_t* composition, float* color
);
CUDA_CALLABLE void get_phong_lighting(
   float* light_direction, float* surface_normal,
   float* reflected_direction, float* viewer_direction,
   composition_t* composition, float* color
);
CUDA_CALLABLE void get_ray_position(
   ray_t* ray, float parameter, float* position
);
CUDA_CALLABLE void get_sphere_normal(
   sphere_t* sphere, float* position, float* normal
);
CUDA_CALLABLE float get_ray_sphere_intersection(ray_t* ray, sphere_t* sphere);

#endif
