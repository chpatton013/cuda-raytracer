#ifndef _UTIL_H_
#define _UTIL_H_

#include <math.h>
#include <stdint.h>
#include <cuda.h>

/**
 * This disgusting function returns a very good approximation of the inverse
 * square root.
 *
 * This function is taken from the Quake III Arena source code.
 * Though the code was edited for style, the content and comments remain
 * unchanged.
 *
 * The basic idea is to interpret the float as an integer (not convert),
 * use a magic number in some math and bitshifting, reinterpret the value
 * as a float, and apply a couple iterations of Newton's method for accuracy.
 */
__host__ __device__ float Q_rsqrt(float number) {
   long i;
   float x2, y;
   float threehalfs = 1.5f;

   x2 = number * 0.5f;
   y = number;
   i = *((long*)&y);                      // evil floating point bit level hacking
   i = 0x5f3759df - ( i >> 1 );           // what the fuck?
   y = *((float*)&i);
   y = y * (threehalfs - (x2 * y * y));   // 1st iteration
   y = y * (threehalfs - (x2 * y * y));   // 2nd iteration, this can be removed

   return y;
}

/**
 * Vector math
 */
__host__ __device__ void zero_vector(float* v) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v[ndx] = 0.0f;
   }
}

__host__ __device__ void copy(float* v_src, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v_src[ndx];
   }
}

__host__ __device__ float dot(float* v1, float* v2) {
   float result = 0.0f;

   for (int ndx = 0; ndx < 3; ++ndx) {
      result += v1[ndx] * v2[ndx];
   }

   return result;
}

__host__ __device__ void cross(float* v1, float* v2, float* v_dest) {
   // Sarrus' Rule for 3x3 determinants
   v_dest[0] = v1[1] * v2[2] - v1[2] * v2[1];
   v_dest[1] = v1[2] * v2[0] - v1[0] * v2[2];
   v_dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

__host__ __device__ float mag(float* v) {
   return 1 / Q_rsqrt(dot(v, v));
}

__host__ __device__ void scale(float* v_src, float s, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v_src[ndx] * s;
   }
}

__host__ __device__ void scale_i(float* v, float s) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v[ndx] *= s;
   }
}

__host__ __device__ void scalev(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] * v2[ndx];
   }
}

__host__ __device__ void scalev_i(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] *= v2[ndx];
   }
}

__host__ __device__ void norm(float* v_src, float* v_dest) {
   scale(v_src, 1.0f / mag(v_src), v_dest);
}

__host__ __device__ void norm_i(float* v) {
   scale_i(v, 1.0f / mag(v));
}

__host__ __device__ void add(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] + v2[ndx];
   }
}

__host__ __device__ void add_i(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] += v2[ndx];
   }
}

__host__ __device__ void sub(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] - v2[ndx];
   }
}

__host__ __device__ void sub_i(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] -= v2[ndx];
   }
}

/**
 * Type-specific
 */
__host__ __device__ void get_ambient_lighting(
   composition_t* composition, float* color
) {
   copy(composition->ambient, color);
}

__host__ __device__ void get_diffuse_lighting(
   float* light_direction, float* surface_normal,
   composition_t* composition, float* color
) {
   float L_dot_N = dot(light_direction, surface_normal);

   if (L_dot_N > 0.0f) {
      scale(composition->diffuse, L_dot_N, color);
   } else {
      zero_vector(color);
   }
}

__host__ __device__ void get_specular_lighting(
   float* reflected_direction, float* viewer_direction,
   composition_t* composition, float* color
) {
   float R_dot_V = dot(reflected_direction, viewer_direction);

   if (R_dot_V > 0.0f) {
      scale(composition->specular, pow(R_dot_V, composition->shine), color);
   } else {
      zero_vector(color);
   }
}

__host__ __device__ void get_phong_lighting(
   float* light_direction, float* surface_normal,
   float* reflected_direction, float* viewer_direction,
   composition_t* composition, float* color
) {
   float diffuse[3];
   float specular[3];
   float ambient[3];

   get_ambient_lighting(composition, ambient);
   get_diffuse_lighting(
      light_direction, surface_normal,
      composition, diffuse
   );
   get_specular_lighting(
      reflected_direction, viewer_direction,
      composition, specular
   );

   add(ambient, diffuse, color);
   add_i(color, specular);
}

__host__ __device__ void get_ray_position(
   ray_t* ray, float parameter, float* position
) {
   scale(ray->direction, parameter, position);
   add_i(position, ray->origin);
}

__host__ __device__ void get_sphere_normal(
   sphere_t* sphere, float* position, float* normal
) {
   sub(sphere->center, position, normal);
   norm_i(normal);
}

__host__ __device__ float get_ray_sphere_intersection(
   ray_t* ray, sphere_t* sphere
) {
   float* c = sphere->center;
   float* d = ray->direction;
   float* e = ray->origin;
   float eMinusC[3];
   sub(e, c, eMinusC);

   float r = sphere->radius,
         A = dot(d, d),
         B = 2 * dot(d, eMinusC),
         C = dot(eMinusC, eMinusC) - r * r,
         discriminant = B * B - 4 * A * C;

   if (discriminant < 0.0f) {
      return -1.0f;
   } else if (discriminant > 0.0f) {
      float disc_root = 1 / Q_rsqrt(discriminant),
            plus_solution = (-B + disc_root) / (2 * A),
            minus_solution = (-B - disc_root) / (2 * A);

      if (minus_solution < 0.0f) {
         return plus_solution;
      } else {
         return fmin(plus_solution, minus_solution);
      }
   } else {
      return -B / (2 * A);
   }
}

#endif
