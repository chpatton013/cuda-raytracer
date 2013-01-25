#include "API.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

void API::Prepare() {
   copy(DFLT_CAMERA_POSITION, camera.position);
   copy(DFLT_CAMERA_FORWARD, camera.forward);
   copy(DFLT_CAMERA_UP, camera.up);
   camera.z_near = DFLT_CAMERA_Z_NEAR;
   camera.z_far = DFLT_CAMERA_Z_FAR;

   light_count = 0;
   sphere_count = 0;
}

uint16_t API::AddLight(Light& light) {
   if (light_count < MAX_LIGHTS) {
      light_arr[light_count] = light;
      return ++light_count;
   } else {
      return 0;
   }
}
uint16_t API::AddSphere(Sphere& sphere) {
   if (sphere_count < MAX_SPHERES) {
      sphere_arr[sphere_count] = sphere;
      return ++sphere_count;
   } else {
      return 0;
   }
}

void API::Draw(uint16_t width, uint16_t height, float hfov) {
   image = Image::MakeImage(width, height);

   float vfov = hfov * height / (float)width;

   uint64_t rays_cast = 0;
   for (uint16_t x = 0; x < width; ++x) {
      for (uint16_t y = 0; y < height; ++y) {
         float scaled_camera_right[3];
         cross(camera.forward, camera.up, scaled_camera_right);
         scalei(
            scaled_camera_right,
            tan(hfov * 0.5f) * (((x + 0.5f) * 2.0f / (float)width) - 1.0f)
         );

         float scaled_camera_up[3];
         scale(
            camera.up,
            tan(vfov * 0.5f) *
               (1.0f - ((height - y + 0.5f) / (float)height) * 2.0f),
            scaled_camera_up
         );

         // ray_direction = normalized(
         //    camera forward + scaled_camera_right + scaled_camera_up
         // )
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

      if (x % (width / 25) == 0) {
         printf("%d%% complete...\n", x * 100 / width);
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

void API::WriteTGA(const char* path) {
   FILE *fp = fopen(path, "w");
   assert(fp);

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

   putc(image.width & 0xff, fp); // width, low byte
   putc((image.width & 0xff00) >> 8, fp); // width, high byte

   putc(image.height & 0xff, fp); // height, low byte
   putc((image.height & 0xff00) >> 8, fp); // height, high byte

   putc(24, fp); // 24-bit color depth

   putc(0, fp);

   // write the raw pixel data in groups of 3 bytes (BGR order)
   for (int y = 0; y < image.height; ++y) {
      for (int x = 0; x < image.width; ++x) {
         unsigned char rbyte, gbyte, bbyte;

         double r = fmin(1.0f, image.Pixel(x, y)[0]);
         double g = fmin(1.0f, image.Pixel(x, y)[1]);
         double b = fmin(1.0f, image.Pixel(x, y)[2]);

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
