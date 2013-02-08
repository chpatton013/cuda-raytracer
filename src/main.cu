#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <tclap/CmdLine.h>
#include "Type.h"
#include "Scene.h"
#include "RayTracer.h"

bool parse_cmd_line(
      int argc, char** argv,
      uint16_t* win_w, uint16_t* win_h,
      std::string* camera_filename,
      std::string* light_filename,
      std::string* geometry_filename,
      std::string* output_filename
);

int main(int argc, char** argv) {
   uint16_t win_w;
   uint16_t win_h;
   std::string camera_filename;
   std::string light_filename;
   std::string geometry_filename;
   std::string output_filename;

   camera_t camera;
   std::vector<light_t> light_vec;
   std::vector<sphere_t> sphere_vec;
   float* img_buffer;

   if (!parse_cmd_line(
         argc, argv,
         &win_w, &win_h,
         &camera_filename,
         &light_filename,
         &geometry_filename,
         &output_filename
   )) {
      return EXIT_FAILURE;
   }
   if (!create_scene(
         &camera, camera_filename,
         &light_vec, light_filename,
         &sphere_vec, geometry_filename,
         win_w, win_h
   )) {
      return EXIT_FAILURE;
   }

   img_buffer = (float*)malloc(sizeof(float) * win_w * win_h * 3);
   draw_scene(
      &light_vec.front(), light_vec.size(),
      &sphere_vec.front(), sphere_vec.size(),
      &camera, img_buffer,
      win_w, win_h
   );
   write_tga(img_buffer, win_w, win_h, output_filename);
   free(img_buffer);

   return EXIT_SUCCESS;
}

bool parse_cmd_line(
      int argc, char** argv,
      uint16_t* win_w, uint16_t* win_h,
      std::string* camera_filename,
      std::string* light_filename,
      std::string* geometry_filename,
      std::string* output_filename
) {
   uint16_t max_win_w = 12800;
   uint16_t max_win_h = 12800;
   uint16_t dflt_win_w = 1280;
   uint16_t dflt_win_h = 720;
   std::string dflt_camera_filename ="camera.txt";
   std::string dflt_light_filename ="lights.txt";
   std::string dflt_geometry_filename ="geometry.txt";
   std::string dflt_output_filename ="output.tga";

   int32_t width_val;
   int32_t height_val;
   std::string camera_val;
   std::string light_val;
   std::string geometry_val;
   std::string output_val;

   try {
      TCLAP::CmdLine cmd("Awesome Ray Tracer", ' ', "", false);

      TCLAP::ValueArg<int32_t> width_arg(
         "w", "width", "Output image width",
         false, dflt_win_w, "positive integer"
      );
      TCLAP::ValueArg<int32_t> height_arg(
         "h", "height", "Output image height",
         false, dflt_win_h, "positive integer"
      );
      TCLAP::ValueArg<std::string> camera_arg(
         "c", "camera", "Filename to use to generate the camera",
         false, dflt_camera_filename, "string"
      );
      TCLAP::ValueArg<std::string> light_arg(
         "l", "light", "Filename to use to generate lights",
         false, dflt_light_filename, "string"
      );
      TCLAP::ValueArg<std::string> geometry_arg(
         "g", "geometry", "Filename to use to generate geometry",
         false, dflt_geometry_filename, "string"
      );
      TCLAP::ValueArg<std::string> output_arg(
         "o", "output", "Filename of output image",
         false, dflt_output_filename, "string"
      );

      cmd.add(width_arg);
      cmd.add(height_arg);
      cmd.add(camera_arg);
      cmd.add(light_arg);
      cmd.add(geometry_arg);
      cmd.add(output_arg);

      cmd.parse(argc, argv);

      width_val = width_arg.getValue();
      height_val = height_arg.getValue();
      camera_val = camera_arg.getValue();
      light_val = light_arg.getValue();
      geometry_val = geometry_arg.getValue();
      output_val = output_arg.getValue();
   } catch (TCLAP::ArgException& e) {
      fprintf(stderr,
         "error: %s for arg %s\n", e.error().c_str(), e.argId().c_str()
      );
      return false;
   }

   // check bounds on window width
   if (width_val > max_win_w) {
      fprintf(stderr,
         "error: window width must be less than %d\n", max_win_w
      );
      return false;
   } else if (width_val <= 0) {
      fprintf(stderr, "error: window width must be positive\n");
      return false;
   } else {
      *win_w = width_val;
   }

   // check bounds on window height
   if (height_val > max_win_h) {
      fprintf(stderr,
         "error: window height must be less than %d\n", max_win_h
      );
      return false;
   } else if (height_val <= 0) {
      fprintf(stderr, "error: window height must be positive\n");
      return false;
   } else {
      *win_h = height_val;
   }

   // assume files exist and are readable
   *camera_filename = camera_val;
   *light_filename = light_val;
   *geometry_filename = geometry_val;
   *output_filename = output_val;


   return true;
}
