#include "API.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>

#include <tclap/CmdLine.h>

void API::Prepare() {
   light_count = 0;
   sphere_count = 0;

   AddCamera(camera_filename);
   AddLights(light_filename);
   AddSpheres(geometry_filename);
}
bool API::ParseArgs(int argc, char** argv) {
   int32_t width_val;
   int32_t height_val;
   std::string camera_val;
   std::string light_val;
   std::string geometry_val;
   std::string output_val;
   bool cuda_val;

   try {
      TCLAP::CmdLine cmd("Awesome Ray Tracer", ' ', "", false);

      TCLAP::ValueArg<int32_t> width_arg(
         "w", "width", "Output image width",
         false, DFLT_IMAGE_WIDTH, "positive integer"
      );
      TCLAP::ValueArg<int32_t> height_arg(
         "h", "height", "Output image height",
         false, DFLT_IMAGE_HEIGHT, "positive integer"
      );
      TCLAP::ValueArg<std::string> camera_arg(
         "c", "camera", "Filename to use to generate the camera",
         false, DFLT_CAMERA_FILENAME, "string"
      );
      TCLAP::ValueArg<std::string> light_arg(
         "l", "light", "Filename to use to generate lights",
         false, DFLT_LIGHT_FILENAME, "string"
      );
      TCLAP::ValueArg<std::string> geometry_arg(
         "g", "geometry", "Filename to use to generate geometry",
         false, DFLT_GEOMETRY_FILENAME, "string"
      );
      TCLAP::ValueArg<std::string> output_arg(
         "o", "output", "Filename of output image",
         false, DFLT_OUTPUT_FILENAME, "string"
      );
      TCLAP::SwitchArg cuda_switch(
         "p", "parallel", "Use CUDA for massive parallelization",
         DFLT_CUDA_MODE
      );

      cmd.add(width_arg);
      cmd.add(height_arg);
      cmd.add(camera_arg);
      cmd.add(light_arg);
      cmd.add(geometry_arg);
      cmd.add(output_arg);
      cmd.add(cuda_switch);

      cmd.parse(argc, argv);

      width_val = width_arg.getValue();
      height_val = height_arg.getValue();
      camera_val = camera_arg.getValue();
      light_val = light_arg.getValue();
      geometry_val = geometry_arg.getValue();
      output_val = output_arg.getValue();
      cuda_val = cuda_switch.getValue();
   } catch (TCLAP::ArgException &e) {
      fprintf(stderr,
         "error: %s for arg %s\n", e.error().c_str(), e.argId().c_str()
      );
      return false;
   }

   // check bounds on image width
   if (width_val > MAX_IMAGE_WIDTH) {
      fprintf(stderr,
         "error: image width must be less than %d\n", MAX_IMAGE_WIDTH
      );
      return false;
   } else if (width_val <= 0) {
      fprintf(stderr, "error: image width must be positive\n");
      return false;
   } else {
      image_width = width_val;
   }

   // check bounds on image height
   if (height_val > MAX_IMAGE_HEIGHT) {
      fprintf(stderr,
         "error: image height must be less than %d\n", MAX_IMAGE_HEIGHT
      );
      return false;
   } else if (height_val <= 0) {
      fprintf(stderr, "error: image height must be positive\n");
      return false;
   } else {
      image_height = height_val;
   }

   // assume files exist and are readable
   camera_filename = camera_val;
   light_filename = light_val;
   geometry_filename = geometry_val;
   output_filename = output_val;

   cuda_mode = cuda_val;

   return true;
}

bool API::AddCamera(const std::string& filename) {
   float p[3];
   float f[3];
   float u[3];
   float zn;
   float zf;
   float h;

   std::ifstream filestream(filename.c_str());
   bool success =
      get_next_vector(filestream, p) &&
      get_next_vector(filestream, f) &&
      get_next_vector(filestream, u) &&
      !isnan(zn = get_next_float(filestream)) &&
      !isnan(zf = get_next_float(filestream)) &&
      !isnan(h = get_next_float(filestream));

   if (success) {
      camera = Camera::MakeCamera(p, f, u, zn, zf, h);
   }

   return success;
}

bool API::AddLights(const std::string& filename) {
   std::ifstream filestream(filename.c_str());
   bool success;
   Light light;

   do {
      float p[3];
      float c[3];

      success = get_next_vector(filestream, p) &&
                get_next_vector(filestream, c);

      if (success) {
         light = Light::MakeLight(p, c);
      }
   } while (success && AddLight(light));

   return success;
}
bool API::AddLight(Light& light) {
   if (light_count < MAX_LIGHTS) {
      light_arr[light_count++] = light;
      return true;
   } else {
      return false;
   }
}

bool API::AddSpheres(const std::string& filename) {
   std::ifstream filestream(filename.c_str());
   bool success;
   Sphere sphere;

   do {
      float e[3];
      float r;
      float a[3];
      float d[3];
      float sc[3];
      float sp;
      Composition c;

      success =
         get_next_vector(filestream, e) &&
         !isnan(r = get_next_float(filestream)) &&
         get_next_vector(filestream, a) &&
         get_next_vector(filestream, d) &&
         get_next_vector(filestream, sc) &&
         !isnan(sp = get_next_float(filestream));

      if (success) {
         c = Composition::MakeComposition(a, d, sc, sp);
         sphere = Sphere::MakeSphere(e, r, c);
      }
   } while (success && AddSphere(sphere));

   return success;
}
bool API::AddSphere(Sphere& sphere) {
   if (sphere_count < MAX_SPHERES) {
      sphere_arr[sphere_count++] = sphere;
      return true;
   } else {
      return 0;
   }
}

void API::Draw() {
   image = Image::MakeImage(image_width, image_height);

   float vfov = camera.hfov * image_height / (float)image_width;
   
   cudaWrapper(image_width, image_height);

   uint64_t rays_cast = 0;
   for (uint16_t x = 0; x < image_width; ++x) {
      for (uint16_t y = 0; y < image_height; ++y) {
         float scaled_camera_right[3];
         cross(camera.forward, camera.up, scaled_camera_right);
         scalei(
            scaled_camera_right,
            tan(camera.hfov * 0.5f) *
             (((x + 0.5f) * 2.0f / (float)image_width) - 1.0f)
         );

         float scaled_camera_up[3];
         scale(
            camera.up,
            tan(vfov * 0.5f) *
             (1.0f - ((image_height - y + 0.5f) / (float)image_height) * 2.0f),
            scaled_camera_up
         );

         float ray_direction[3];
         add(zero_vector, camera.forward, ray_direction);
         addi(ray_direction, scaled_camera_right);
         addi(ray_direction, scaled_camera_up);
         normi(ray_direction);

         ++rays_cast;
         Ray ray = Ray::MakeRay(camera.position, ray_direction);
         Sphere* sphere;

         float color[3];
         copy(zero_vector, color);
         float intersection = GetClosestIntersection(&ray, &sphere);
         if (intersection >= 0.0f) {
            LightSurface(&ray, intersection, sphere, color);
         }
         image.Pixel(x, y, color);
      }

      if (x % (image_width / 25) == 0) {
         float complete = ((x + 25) / (float)image_width);
         printf("%d%% complete...\n", (int)(complete * 100.0f));
      }
   }

   printf("DONE: %llu rays cast\n", (long long unsigned int)rays_cast);
}
float API::GetClosestIntersection(Ray* ray, Sphere** sphere) {
   float closest_param = FLT_MAX;
   Sphere* closest_sphere = NULL;

   for (int ndx = 0; ndx < sphere_count; ++ndx) {
      float current_param = sphere_arr[ndx].Intersect(ray);

      if (current_param >= 0.0f && current_param < closest_param) {
         closest_param = current_param;
         closest_sphere = sphere_arr + ndx;
      }
   }

   if (closest_sphere == NULL) {
      return -1.0f;
   } else {
      *sphere = closest_sphere;
      return closest_param;
   }
}
void API::LightSurface(
   Ray* ray, float parameter, Sphere* sphere, float* color
) {
   float position[3] = {0.0f, 0.0f, 0.0f};
   float normal[3] = {0.0f, 0.0f, 0.0f};
   float viewer[3] = {0.0f, 0.0f, 0.0f};

   // position = ray position(parameter)
   ray->Position(parameter, position);
   // normal = sphere normal(position)
   sphere->Normal(position, normal);
   // viewer = normalized(position - camera position)
   sub(position, camera.position, viewer);
   normi(viewer);

   for (int ndx = 0; ndx < light_count; ++ndx) {
      float light_direction[3];
      float reflected[3];

      // light_direction = normalized(position - light position)
      sub(position, light_arr[ndx].position, light_direction);
      normi(light_direction);

      // reflected = normalized(
      //    (2 * (light_direction dot normal) * normal) - light_direction
      // )
      scale(normal, (2 * dot(light_direction, normal)), reflected);
      subi(reflected, light_direction);
      normi(reflected);

      float diffuse[3];
      float specular[3];
      float ambient[3];

      sphere->comp.DiffuseLighting(light_direction, normal, diffuse);
      sphere->comp.SpecularLighting(reflected, viewer, specular);
      sphere->comp.AmbientLighting(ambient);

      addi(color, diffuse);
      addi(color, specular);
      addi(color, ambient);
   }
}

void API::WriteTGA() {
   image.WriteTGA(output_filename.c_str());
}
