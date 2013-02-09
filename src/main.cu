#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <tclap/CmdLine.h>
#include "Timer.h"
#include "Type.h"
#include "Scene.h"

// Externs
extern GLuint pbo_handle;
extern GLuint texture_handle;
extern bool initialize_cuda(
   light_t* lights, uint16_t light_count,
   sphere_t* spheres, uint16_t sphere_count,
   camera_t* camera, uint16_t win_w, uint16_t win_h
);
extern void cleanup_cuda();
extern void draw_scene();

// Globals
uint16_t win_w;
uint16_t win_h;
static std::string camera_filename;
static std::string light_filename;
static std::string geometry_filename;

// Values
static camera_t camera;
static std::vector<light_t> light_vec;
static std::vector<sphere_t> sphere_vec;

// Prototypes
bool parse_cmd_line(
      int argc, char** argv,
      uint16_t* win_w, uint16_t* win_h,
      std::string* camera_filename,
      std::string* light_filename,
      std::string* geometry_filename
);
bool initialize_gl(int argc, char** argv, uint16_t win_w, uint16_t win_h);
void display();
void display_texture(
   GLuint* pbo, GLuint* texture, uint16_t win_w, uint16_t win_h
);
void display_fps(Timer* timer);
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

int main(int argc, char** argv) {
   if (!parse_cmd_line(
         argc, argv,
         &win_w, &win_h,
         &camera_filename,
         &light_filename,
         &geometry_filename
   )) {
      return EXIT_FAILURE;
   }
   if (!initialize_gl(argc, argv, win_w, win_h)) {
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
   if (!initialize_cuda(
      &light_vec.front(), light_vec.size(),
      &sphere_vec.front(), sphere_vec.size(),
      &camera, win_w, win_h
   )) {
      return EXIT_FAILURE;
   }

   glutDisplayFunc(display);
   glutKeyboardFunc(keyboard);
   glutMouseFunc(mouse);
   glutMotionFunc(motion);

   glutMainLoop();

   cleanup_cuda();

   return EXIT_SUCCESS;
}

bool parse_cmd_line(
      int argc, char** argv,
      uint16_t* win_w, uint16_t* win_h,
      std::string* camera_filename,
      std::string* light_filename,
      std::string* geometry_filename
) {
   uint16_t max_win_w = 12800;
   uint16_t max_win_h = 12800;
   uint16_t dflt_win_w = 1280;
   uint16_t dflt_win_h = 720;
   std::string dflt_camera_filename ="camera.txt";
   std::string dflt_light_filename ="lights.txt";
   std::string dflt_geometry_filename ="geometry.txt";

   int32_t width_val;
   int32_t height_val;
   std::string camera_val;
   std::string light_val;
   std::string geometry_val;

   try {
      TCLAP::CmdLine cmd("Awesome Ray Tracer", ' ', "", false);

      TCLAP::ValueArg<int32_t> width_arg(
         "w", "width", "Window width",
         false, dflt_win_w, "positive integer"
      );
      TCLAP::ValueArg<int32_t> height_arg(
         "h", "height", "Window height",
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

      cmd.add(width_arg);
      cmd.add(height_arg);
      cmd.add(camera_arg);
      cmd.add(light_arg);
      cmd.add(geometry_arg);

      cmd.parse(argc, argv);

      width_val = width_arg.getValue();
      height_val = height_arg.getValue();
      camera_val = camera_arg.getValue();
      light_val = light_arg.getValue();
      geometry_val = geometry_arg.getValue();
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

   return true;
}

bool initialize_gl(int argc, char** argv, uint16_t win_w, uint16_t win_h) {
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

   glutInitWindowSize(win_w, win_h);
   glutCreateWindow("");
   glViewport(0, 0, win_w, win_h);

   glClearColor(0.0, 0.0, 0.0, 1.0);
   glDisable(GL_DEPTH_TEST);

   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);

   return true;
}

void display() {
   static Timer timer;
   timer.start();

   draw_scene();
   display_texture(&pbo_handle, &texture_handle, win_w, win_h);

   timer.stop();
   display_fps(&timer);

   glutSwapBuffers();
   glutPostRedisplay();
}

void display_texture(
   GLuint* pbo, GLuint* texture, uint16_t win_w, uint16_t win_h
) {
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_handle);
   glBindTexture(GL_TEXTURE_2D, texture_handle);

   glTexSubImage2D(
      GL_TEXTURE_2D, 0, 0,
      0, win_w, win_h,
      GL_RGBA, GL_UNSIGNED_BYTE, NULL
   );

   glBegin(GL_QUADS);

   glTexCoord2f(0.0f, 1.0f);
   glVertex3f(0.0f, 0.0f, 0.0f);

   glTexCoord2f(0.0f, 0.0f);
   glVertex3f(0.0f, 1.0f, 0.0f);

   glTexCoord2f(1.0f, 0.0f);
   glVertex3f(1.0f, 1.0f, 0.0f);

   glTexCoord2f(1.0f, 1.0f);
   glVertex3f(1.0f, 0.0f, 0.0f);

   glEnd();
}

void display_fps(Timer* timer) {
   static int threshold = 100;
   static int counter = threshold;

   if (counter == threshold) {
      static const int ONE_BILLION = 1000000000;
      static const int ONE_MILLION = 1000000;

      uint64_t duration = timer->get();
      double frame_rate = ONE_BILLION / (double)duration;

      char str[256];
      sprintf(
         str, "Awesome Ray Tracer: %3.4lf FPS - %3.4f MS",
         frame_rate, duration / (double)ONE_MILLION
      );

      counter = 0;
   }

   ++counter;
}

void keyboard(unsigned char key, int x, int y) {
   glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
   glutPostRedisplay();
}

void motion(int x, int y) {
   glutPostRedisplay();
}
