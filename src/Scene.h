#ifndef _SCENE_H_
#define _SCENE_H_

#include <stdint.h>
#include <string>
#include <vector>
#include "Type.h"

void create_camera(
      camera_t* camera, std::string filename,
      uint16_t win_w, uint16_t win_h
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
      std::vector<sphere_t>* sphere_vec, std::string sphere_filename,
      uint16_t win_w, uint16_t win_h
);

void get_center_of_scene(camera_t* camera, float* CoS);

float get_next_float(std::ifstream& filestream);
bool get_next_vector(std::ifstream& filestream, float* buffer);

#endif
