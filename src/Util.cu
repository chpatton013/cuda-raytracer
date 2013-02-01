#include "Util.h"

#include <math.h>
#include <stdint.h>

/**
 * This disgusting function returns a very good approximation of the inverse
 * square root.
 *
 * This function is taken from the Quake III Arena source code.
 * Though the code was edited for style, the content and comments remain
 * unchanged.
 */
float Q_rsqrt(float number) {
   long i;
   float x2, y;
   float threehalfs = 1.5f;

   x2 = number * 0.5f;
   y = number;
   i = *((long*)&y);                      // evil floating point bit level hacking
   i = 0x5f3759df - ( i >> 1 );           // what the fuck?
   y = *((float*)&i);
   y = y * (threehalfs - (x2 * y * y));   // 1st iteration
   y = y * (threehalfs - (x2 * y * y));   // 2nd iteration, this can be removed

   return y;
}

void zero_vector(float* v) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v[ndx] = 0.0f;
   }
}
void copy(float* v_src, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v_src[ndx];
   }
}
float dot(float* v1, float* v2) {
   float result = 0.0f;

   for (int ndx = 0; ndx < 3; ++ndx) {
      result += v1[ndx] * v2[ndx];
   }

   return result;
}
void cross(float* v1, float* v2, float* v_dest) {
   // Sarrus' Rule for 3x3 determinants
   v_dest[0] = v1[1] * v2[2] - v1[2] * v2[1];
   v_dest[1] = v1[2] * v2[0] - v1[0] * v2[2];
   v_dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
}
float mag(float* v) {
   return 1 / Q_rsqrt(dot(v, v));
}
void norm(float* v_src, float* v_dest) {
   scale(v_src, 1.0f / mag(v_src), v_dest);
}
void norm_i(float* v) {
   scale_i(v, 1.0f / mag(v));
}
void add(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] + v2[ndx];
   }
}
void add_i(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] += v2[ndx];
   }
}
void sub(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] - v2[ndx];
   }
}
void sub_i(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] -= v2[ndx];
   }
}
void scale(float* v_src, float s, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v_src[ndx] * s;
   }
}
void scale_i(float* v, float s) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v[ndx] *= s;
   }
}
void scalev(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] * v2[ndx];
   }
}
void scalev_i(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] *= v2[ndx];
   }
}
