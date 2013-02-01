#include "main.h"

#include <stdio.h>
#include <stdlib.h>
#include <tclap/CmdLine.h>

int main(int argc, char** argv) {
   if(!parse_cmd_line(argc, argv)){
      return EXIT_FAILURE;
   }

   create_scene(
      &camera, camera_filename,
      &light_vec, light_filename,
      &sphere_vec, geometry_filename
   );
   img_buffer = (float*)malloc(sizeof(float) * img_w * img_h * 3);
   draw_scene(
      &light_vec.front(), light_vec.size(),
      &sphere_vec.front(), sphere_vec.size(),
      &camera, img_buffer,
      img_w, img_h,
      cpu_mode
   );
   write_tga(img_buffer, img_w, img_h, output_filename);
   free(img_buffer);

   return EXIT_SUCCESS;
}

bool parse_cmd_line(int argc, char** argv) {
   int32_t width_val;
   int32_t height_val;
   std::string camera_val;
   std::string light_val;
   std::string geometry_val;
   std::string output_val;
   bool cpu_val;

   try {
      TCLAP::CmdLine cmd("Awesome Ray Tracer", ' ', "", false);

      TCLAP::ValueArg<int32_t> width_arg(
         "w", "width", "Output image width",
         false, dflt_img_w, "positive integer"
      );
      TCLAP::ValueArg<int32_t> height_arg(
         "h", "height", "Output image height",
         false, dflt_img_h, "positive integer"
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
      TCLAP::SwitchArg cpu_switch(
         "s", "cpu", "Use the CPU for image processing",
         dflt_cpu_mode
      );

      cmd.add(width_arg);
      cmd.add(height_arg);
      cmd.add(camera_arg);
      cmd.add(light_arg);
      cmd.add(geometry_arg);
      cmd.add(output_arg);
      cmd.add(cpu_switch);

      cmd.parse(argc, argv);

      width_val = width_arg.getValue();
      height_val = height_arg.getValue();
      camera_val = camera_arg.getValue();
      light_val = light_arg.getValue();
      geometry_val = geometry_arg.getValue();
      output_val = output_arg.getValue();
      cpu_val = cpu_switch.getValue();
   } catch (TCLAP::ArgException& e) {
      fprintf(stderr,
         "error: %s for arg %s\n", e.error().c_str(), e.argId().c_str()
      );
      return false;
   }

   // check bounds on image width
   if (width_val > max_img_w) {
      fprintf(stderr,
         "error: image width must be less than %d\n", max_img_w
      );
      return false;
   } else if (width_val <= 0) {
      fprintf(stderr, "error: image width must be positive\n");
      return false;
   } else {
      img_w = width_val;
   }

   // check bounds on image height
   if (height_val > max_img_h) {
      fprintf(stderr,
         "error: image height must be less than %d\n", max_img_h
      );
      return false;
   } else if (height_val <= 0) {
      fprintf(stderr, "error: image height must be positive\n");
      return false;
   } else {
      img_h = height_val;
   }

   // assume files exist and are readable
   camera_filename = camera_val;
   light_filename = light_val;
   geometry_filename = geometry_val;
   output_filename = output_val;

   cpu_mode = cpu_val;

   return true;
}
