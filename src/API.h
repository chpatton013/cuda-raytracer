#ifndef _API_H_
#define _API_H_

#include <stdint.h>

#include <string>

#include "Type.h"
#include "Util.h"

namespace API {
   using namespace Util;
   using namespace Type;

   // Limits
   static const uint16_t MAX_IMAGE_WIDTH = 65535;
   static const uint16_t MAX_IMAGE_HEIGHT = 65535;
   static const uint16_t MAX_LIGHTS = 8;
   static const uint16_t MAX_SPHERES = 32768;

   // Defaults
   static const uint16_t DFLT_IMAGE_WIDTH = 1280;
   static const uint16_t DFLT_IMAGE_HEIGHT = 720;
   static const std::string DFLT_CAMERA_FILENAME = "camera.txt";
   static const std::string DFLT_LIGHT_FILENAME = "lights.txt";
   static const std::string DFLT_GEOMETRY_FILENAME = "geometry.txt";
   static const std::string DFLT_OUTPUT_FILENAME = "output.tga";
   static const bool DFLT_CUDA_MODE = false;

   // Values
   static uint16_t image_width;
   static uint16_t image_height;
   static std::string camera_filename;
   static std::string light_filename;
   static std::string geometry_filename;
   static std::string output_filename;
   static bool cuda_mode;

   static uint16_t light_count;
   static uint16_t sphere_count;

   static Camera camera;
   static Light light_arr[MAX_LIGHTS];
   static Sphere sphere_arr[MAX_SPHERES];
   static Image image;

   // Functions
   void Prepare();
   bool ParseArgs(int argc, char** argv);

   bool AddCamera(const std::string& filename);

   bool AddLights(const std::string& filename);
   bool AddLight(Light& light);

   bool AddSpheres(const std::string& filename);
   bool AddSphere(Sphere& sphere);

   void Draw();
   float GetClosestIntersection(Ray* ray, Sphere** sphere);
   void LightSurface(Ray* ray, float parameter, Sphere* sphere, float* color);

   void WriteTGA();
}

#endif
