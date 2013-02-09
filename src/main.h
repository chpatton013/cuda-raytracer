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
static std::vector<light_t> light_vec;
static std::vector<sphere_t> sphere_vec;
unsigned int window_width;
unsigned int window_height;
camera_t camera;
sphere_t* spheres;
light_t* lights;
uint16_t sphere_count;
uint16_t light_count;
camera_t* d_camera;
sphere_t* d_spheres;
light_t* d_lights;

bool parse_cmd_line(int argc, char** argv);

extern bool initGL(int argc, char** argv);
extern void initCuda();
extern void fpsDisplay();
extern void keyboard(unsigned char key, int x, int y);
extern void mouse(int button, int state, int x, int y);
extern void motion(int x, int y);

#endif
