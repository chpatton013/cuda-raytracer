#include <math.h>
#include <stdlib.h>

#include "Type.h"
#include "API.h"

using namespace Type;

int main(int argc, char** argv) {
   // TODO: command line parsing

   API::Prepare();

   // TODO: generify scene creation
   // create lights
   float light_position[3] = {-5.0f, 5.0f, 5.0f};
   float light_color[3] = {1.0f, 1.0f, 1.0f};
   Light light = Light::MakeLight(light_position, light_color);

   API::AddLight(light);

   // create geometry
   float ambient[3] = {0.05f, 0.05f, 0.05f};
   float diffuse[3] = {1.0f, 0.0f, 1.0f};
   float specular[3] = {1.0f, 1.0f, 1.0f};
   float spec_pow = 16.0f;
   Composition sphere_composition = Composition::MakeComposition(
      ambient, diffuse, specular, spec_pow
   );

   float sphere_position[3] = {0.0f, 0.0f, 0.0f};
   float sphere_radius = 1;
   Sphere sphere = Sphere::MakeSphere(
      sphere_position, sphere_radius, sphere_composition
   );
   // END generify

   API::AddSphere(sphere);

   API::Draw(400, 225, 35.0f * M_PI / 180.0f);

   API::WriteTGA("./output.tga");

   return EXIT_SUCCESS;
}
