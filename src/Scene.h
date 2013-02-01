#ifndef _SCENE_H_
#define _SCENE_H_

#include <string>
#include <vector>
#include "Type.h"

void create_camera(
      camera_t* camera, std::string filename
);
void create_lights(
      std::vector<light_t>* light_vec, std::string filename
);
void create_spheres(
      std::vector<sphere_t>* sphere_vec, std::string filename
);
void create_scene(
      camera_t* camera, std::string camera_filename,
      std::vector<light_t>* light_vec, std::string light_filename,
      std::vector<sphere_t>* sphere_vec, std::string sphere_filename
);

float get_next_float(std::ifstream& filestream);
bool get_next_vector(std::ifstream& filestream, float* buffer);

void write_tga(float* img_buffer, int img_w, int img_h, std::string filename);

#endif
