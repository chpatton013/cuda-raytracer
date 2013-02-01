#include "Type.h"

#include <math.h>

void get_ambient_lighting(composition_t* composition, float* color) {
   copy(composition->ambient, color);
}

void get_diffuse_lighting(
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

void get_specular_lighting(
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

void get_phong_lighting(
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

void get_ray_position(ray_t* ray, float parameter, float* position) {
   scale(ray->direction, parameter, position);
   add_i(position, ray->origin);
}

void get_sphere_normal(sphere_t* sphere, float* position, float* normal) {
   sub(sphere->center, position, normal);
   norm_i(normal);
}

float get_ray_sphere_intersection(ray_t* ray, sphere_t* sphere) {
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
