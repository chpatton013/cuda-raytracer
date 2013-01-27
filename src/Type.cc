#include "Type.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Type::Camera Type::Camera::MakeCamera(
   float* p, float* f, float* u, float zn, float zf, float h
) {
   Camera camera;

   copy(p, camera.position);
   copy(f, camera.forward);
   copy(u, camera.up);
   camera.z_near = zn;
   camera.z_far = zf;
   camera.hfov = h * M_PI / 180.0; // given in degrees, needed in radians

   return camera;
}
void Type::Camera::print() {
   Util::print(position, true);
   Util::print(forward, true);
   Util::print(up, true);
   Util::print(z_near, true);
   Util::print(z_far, true);
   Util::print(hfov, true);
}

Type::Light Type::Light::MakeLight(float* p, float* c) {
   Light light;

   copy(p, light.position);
   copy(c, light.color);

   return light;
}
void Type::Light::print() {
   Util::print(position, true);
   Util::print(color, true);
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
void Type::Composition::print() {
   Util::print(ambient, true);
   Util::print(diffuse, true);
   Util::print(specular, true);
   Util::print(spec_pow, true);
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
void Type::Sphere::print() {
   Util::print(center, true);
   Util::print(radius, true);
   comp.print();
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
void Type::Image::WriteTGA(const char* path) {
   FILE *fp = fopen(path, "w");
   assert(fp);

   // write 24-bit uncompressed targa header
   // thanks to Paul Bourke (http://local.wasp.uwa.edu.au/~pbourke/dataformats/tga/)
   putc(0, fp);
   putc(0, fp);

   putc(2, fp); // type is uncompressed RGB

   putc(0, fp);
   putc(0, fp);
   putc(0, fp);
   putc(0, fp);
   putc(0, fp);

   putc(0, fp); // x origin, low byte
   putc(0, fp); // x origin, high byte

   putc(0, fp); // y origin, low byte
   putc(0, fp); // y origin, high byte

   putc(width & 0xff, fp); // width, low byte
   putc((width & 0xff00) >> 8, fp); // width, high byte

   putc(height & 0xff, fp); // height, low byte
   putc((height & 0xff00) >> 8, fp); // height, high byte

   putc(24, fp); // 24-bit color depth

   putc(0, fp);

   // write the raw pixel data in groups of 3 bytes (BGR order)
   for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
         unsigned char rbyte, gbyte, bbyte;

         double r = fmin(1.0f, Pixel(x, y)[0]);
         double g = fmin(1.0f, Pixel(x, y)[1]);
         double b = fmin(1.0f, Pixel(x, y)[2]);

         rbyte = (unsigned char)(r * 255);
         gbyte = (unsigned char)(g * 255);
         bbyte = (unsigned char)(b * 255);

         putc(bbyte, fp);
         putc(gbyte, fp);
         putc(rbyte, fp);
      }
   }

   fclose(fp);
}
