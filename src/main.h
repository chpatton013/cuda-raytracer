#ifndef _MAIN_H_
#define _MAIN_H_

#include <string>
#include <vector>
#include <math.h>
#include <stdint.h>
#include "Type.h"
#include "Scene.h"
#include "RayTracer.h"

#define TITLE_STR "Awesome Ray Tracer"

// Defaults
static const uint16_t max_img_w = 65535;
static const uint16_t max_img_h = 65535;
static const uint16_t dflt_img_w = 1280;
static const uint16_t dflt_img_h = 720;
static const std::string dflt_camera_filename ="camera.txt";
static const std::string dflt_light_filename ="lights.txt";
static const std::string dflt_geometry_filename ="geometry.txt";

// Globals
static uint16_t img_w;
static uint16_t img_h;
static std::string camera_filename;
static std::string light_filename;
static std::string geometry_filename;

// Scene Data
static camera_t camera;
static std::vector<light_t> light_vec;
static std::vector<sphere_t> sphere_vec;

bool parse_cmd_line(int argc, char** argv);
void create_window(
   int argc, char** argv,
   uint16_t win_w, uint16_t, win_h
);

#endif
