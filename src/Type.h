#ifndef _TYPE_H_
#define _TYPE_H_

struct light_t {
   float position[3];
   float color[3];
};

struct composition_t {
   float ambient[3];
   float diffuse[3];
   float specular[3];
   float shine;
};

struct sphere_t {
   float center[3];
   float radius;
   composition_t composition;
};

struct ray_t {
   float origin[3];
   float direction[3];
};

struct camera_t {
   float position[3];
   float front[3];
   float up[3];
   float z[3]; // near, far, focal
   float fov[2]; // horiz, vert: degrees
};

#endif
