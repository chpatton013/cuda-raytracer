#include "Scene.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

void create_camera(
      camera_t* camera, std::string filename,
      uint16_t win_w, uint16_t win_h
) {
   std::ifstream filestream(filename.c_str());
   bool success =
      get_next_vector(filestream, camera->position) &&
      get_next_vector(filestream, camera->front) &&
      get_next_vector(filestream, camera->up) &&
      get_next_vector(filestream, camera->z) &&
      !isnan(camera->fov[0] = get_next_float(filestream));

   if (!success) {
      fprintf(stderr, "error: failed to create camera\n");
      exit(EXIT_FAILURE);
   }

   camera->fov[0] *= M_PI / 180.0f;
   camera->fov[1] = camera->fov[0] / (win_w / (float)win_h);
}
void create_lights(
      std::vector<light_t>* light_vec, std::string filename
) {
   std::ifstream filestream(filename.c_str());
   bool success;
   light_t light;

   do {
      success = get_next_vector(filestream, light.position) &&
                get_next_vector(filestream, light.color);

      if (success) {
         light_vec->push_back(light);
      }
   } while (success);
}
void create_spheres(
      std::vector<sphere_t>* sphere_vec, std::string filename
) {
   std::ifstream filestream(filename.c_str());
   bool success;
   sphere_t sphere;

   do {
      success =
         get_next_vector(filestream, sphere.center) &&
         !isnan(sphere.radius = get_next_float(filestream)) &&
         get_next_vector(filestream, sphere.composition.ambient) &&
         get_next_vector(filestream, sphere.composition.diffuse) &&
         get_next_vector(filestream, sphere.composition.specular) &&
         !isnan(sphere.composition.shine = get_next_float(filestream));

      if (success) {
         sphere_vec->push_back(sphere);
      }
   } while (success);
}
void create_scene(
      camera_t* camera, std::string camera_filename,
      std::vector<light_t>* light_vec, std::string light_filename,
      std::vector<sphere_t>* sphere_vec, std::string sphere_filename,
      uint16_t win_w, uint16_t win_h
) {
   create_camera(camera, camera_filename, win_w, win_h);
   create_lights(light_vec, light_filename);
   create_spheres(sphere_vec, sphere_filename);
}

void get_center_of_scene(camera_t* camera, float* CoS) {
   float z_mid = (camera->z[1] - camera->z[0]) * 0.5f + camera->z[0];

   CoS[0] = camera->position[0];
   CoS[1] = camera->position[1];
   CoS[2] = camera->position[2] + z_mid * 0.5f;
}

float get_next_float(std::ifstream& filestream) {
   float next_float;

   if (filestream.good()) {
      filestream >> next_float;
      return next_float;
   } else {
      return NAN;
   }
}
bool get_next_vector(std::ifstream& filestream, float* buffer) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      if (isnan(buffer[ndx] = get_next_float(filestream))) {
         return false;
      }
   }

   return true;
}
