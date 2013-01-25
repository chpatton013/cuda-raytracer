#ifndef _API_H_
#define _API_H_

#include <stdint.h>

#include "Util.h"
#include "Type.h"

namespace API {
   using namespace Util;
   using namespace Type;

   static float DFLT_CAMERA_POSITION[3] = {0.0f, 0.0f, 10.0f};
   static float DFLT_CAMERA_FORWARD[3] = {0.0f, 0.0f, -1.0f};
   static float DFLT_CAMERA_UP[3] = {0.0f, 1.0f, 0.0f};
   static float DFLT_CAMERA_Z_NEAR = 0.1f;
   static float DFLT_CAMERA_Z_FAR = 100.0f;

   static const uint16_t MAX_LIGHTS = 8;
   static uint16_t light_count = 0;

   static const uint16_t MAX_SPHERES = 32768;
   static uint16_t sphere_count = 0;

   static Camera camera;
   static Light light_arr[MAX_LIGHTS];
   static Sphere sphere_arr[MAX_SPHERES];
   static Image image;

   void Prepare();

   uint16_t AddLight(Light& light);
   uint16_t AddSphere(Sphere& sphere);

   void Draw(uint16_t width, uint16_t height, float hfov);

   float GetClosestIntersection(Ray* ray, Sphere** sphere);

   void LightSurface(Ray* ray, float parameter, Sphere* sphere, float* color);

   void WriteTGA(const char* path);
}

#endif
