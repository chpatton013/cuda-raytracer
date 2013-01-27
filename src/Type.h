#ifndef _TYPE_H_
#define _TYPE_H_

#include <stdint.h>
#include <stdio.h>

#include <string>

#include "Util.h"

namespace Type {
   using namespace Util;

   struct Camera {
      static Camera MakeCamera(
         float* p, float* f, float* u, float zn, float zf, float h
      );
      void print();

      float position[3];
      float forward[3];
      float up[3];
      float z_near;
      float z_far;
      float hfov;
   };

   struct Light {
      static Light MakeLight(float* p, float* c);
      void print();

      float position[3];
      float color[3];
   };

   struct Composition {
      static Composition MakeComposition(
         float* a, float* d, float* sc, float sp
      );
      void print();

      void AmbientLighting(float* color);
      void DiffuseLighting(float* light_direction, float* normal, float* color);
      void SpecularLighting(float* reflected, float* viewer, float* color);

      float ambient[3];
      float diffuse[3];
      float specular[3];
      float spec_pow;
   };

   struct Ray {
      static Ray MakeRay(float* o, float* d);

      void Position(float parameter, float* position);

      float origin[3];
      float direction[3];
   };

   struct Sphere {
      static Sphere MakeSphere(float* e, float r, Composition& c);
      void print();

      float Intersect(Ray* ray);
      void Normal(float* point, float* normal);

      float center[3];
      float radius;
      Composition comp;
   };

   struct Image {
      static Image MakeImage(uint16_t w, uint16_t h);

      bool Pixel(uint16_t x, uint16_t y, float* color);
      float* Pixel(uint16_t x, uint16_t y);
      void WriteTGA(const char* path);

      float* buffer;
      uint16_t width;
      uint16_t height;
   };
}

#endif
