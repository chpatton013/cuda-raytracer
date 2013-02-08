#ifndef _MAIN_H_
#define _MAIN_H_

#include <string>
#include <vector>
#include <math.h>
#include <stdint.h>
#include "Type.h"
#include "Scene.h"
#include "RayTracer.h"

// Defaults
static const uint16_t max_win_w = 12800;
static const uint16_t max_win_h = 12800;
static const uint16_t dflt_win_w = 1280;
static const uint16_t dflt_win_h = 720;
static const std::string dflt_camera_filename ="camera.txt";
static const std::string dflt_light_filename ="lights.txt";
static const std::string dflt_geometry_filename ="geometry.txt";
static const std::string dflt_output_filename ="output.tga";

// Globals
static uint16_t win_w;
static uint16_t win_h;
static std::string camera_filename;
static std::string light_filename;
static std::string geometry_filename;
static std::string output_filename;

// Scene Data
static camera_t camera;
static std::vector<light_t> light_vec;
static std::vector<sphere_t> sphere_vec;
static float* img_buffer;

bool parse_cmd_line(int argc, char** argv);

#endif
