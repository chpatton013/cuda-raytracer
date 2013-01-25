#include "Type.h"

#include <math.h>
#include <stdlib.h>

Type::Light Type::Light::MakeLight(float* p, float* c) {
   Light light;

   copy(p, light.position);
   copy(c, light.color);

   return light;
}

Type::Composition Type::Composition::MakeComposition (
   float* a, float* d, float* sc, float sp
) {
   Composition comp;

   copy(a, comp.ambient);
   copy(d, comp.diffuse);
   copy(sc, comp.specular);
   comp.spec_pow = sp;

   return comp;
}
void Type::Composition::AmbientLighting(float* color) {
   copy(ambient, color);
}
void Type::Composition::DiffuseLighting(
   float* light_direction, float* normal, float* color
) {
   float L_dot_N = dot(light_direction, normal);

   if (L_dot_N > 0.0f) {
      scale(diffuse, L_dot_N, color);
   } else {
      copy(zero_vector, color);
   }
}
void Type::Composition::SpecularLighting(
   float* reflected, float* viewer, float* color
) {
   float R_dot_V = dot(reflected, viewer);

   if (R_dot_V > 0.0f) {
      scale(specular, pow(R_dot_V, spec_pow), color);
   } else {
      copy(zero_vector, color);
   }
}

Type::Ray Type::Ray::MakeRay(float* o, float* d) {
   Ray ray;

   copy(o, ray.origin);
   copy(d, ray.direction);

   return ray;
}
void Type::Ray::Position(float parameter, float* position) {
   scale(direction, parameter, position);
   addi(position, origin);
}

Type::Sphere Type::Sphere::MakeSphere(float* e, float r, Composition& c) {
   Sphere sphere;

   copy(e, sphere.center);
   sphere.radius = r;
   sphere.comp = c;

   return sphere;
}
float Type::Sphere::Intersect(Type::Ray* ray) {
   float* c = center;
   float* d = ray->direction;
   float* e = ray->origin;
   float eMinusC[3];
   sub(e, c, eMinusC);

   float r = radius,
         A = dot(d, d),
         B = 2 * dot(d, eMinusC),
         C = dot(eMinusC, eMinusC) - r * r,
         discriminant = B * B - 4 * A * C;

   if (discriminant < 0.0f) {
      return -1.0f;
   } else if (discriminant > 0.0f) {
      float disc_root = sqrt(discriminant),
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
void Type::Sphere::Normal(float* position, float* normal) {
   sub(center, position, normal);
   normi(normal);
}

Type::Image Type::Image::MakeImage(uint16_t w, uint16_t h) {
   Image image;

   image.width = w;
   image.height = h;
   image.buffer = (float*)calloc(3 * sizeof(float), w * h);

   return image;
}
bool Type::Image::Pixel(uint16_t x, uint16_t y, float* color) {
   if (x < width && y < height) {
      float* location = buffer + 3 * (y * width + x);
      copy(color, location);
      return true;
   } else {
      return false;
   }
}
float* Type::Image::Pixel(uint16_t x, uint16_t y) {
   if (x < width && y < height) {
      return buffer + 3 * (y * width + x);
   } else {
      return NULL;
   }
}
