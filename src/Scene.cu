#include "Scene.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

void create_camera(
      camera_t* camera, std::string filename
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
      std::vector<sphere_t>* sphere_vec, std::string sphere_filename
) {
   create_camera(camera, camera_filename);
   create_lights(light_vec, light_filename);
   create_spheres(sphere_vec, sphere_filename);
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

void write_tga(float* img_buffer, int img_w, int img_h, std::string filename) {
   FILE *fp = fopen(filename.c_str(), "w");
   if (fp == NULL) {
      fprintf(stderr, "error: failed to open file '%s'\n", filename.c_str());
      exit(EXIT_FAILURE);
   }

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

   putc(img_w & 0xff, fp); // width, low byte
   putc((img_w & 0xff00) >> 8, fp); // width, high byte

   putc(img_h & 0xff, fp); // height, low byte
   putc((img_h & 0xff00) >> 8, fp); // height, high byte

   putc(24, fp); // 24-bit color depth

   putc(0, fp);

   // write the raw pixel data in groups of 3 bytes (BGR order)
   for (int y = 0; y < img_h; ++y) {
      for (int x = 0; x < img_w; ++x) {
         unsigned char rbyte, gbyte, bbyte;

         float r = std::min(1.0f, (img_buffer + 3 * (y * img_w + x))[0]);
         float g = std::min(1.0f, (img_buffer + 3 * (y * img_w + x))[1]);
         float b = std::min(1.0f, (img_buffer + 3 * (y * img_w + x))[2]);

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
