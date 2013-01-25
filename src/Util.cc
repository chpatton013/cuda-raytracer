#include "Util.h"

#include <math.h>
#include <stdio.h>

void Util::copy(float* v_src, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v_src[ndx];
   }
}
float Util::dot(float* v1, float* v2) {
   float result = 0.0f;

   for (int ndx = 0; ndx < 3; ++ndx) {
      result += v1[ndx] * v2[ndx];
   }

   return result;
}
void Util::cross(float* v1, float* v2, float* v_dest) {
   // Sarrus' Rule for 3x3 determinants
   v_dest[0] = v1[1] * v2[2] - v1[2] * v2[1];
   v_dest[1] = v1[2] * v2[0] - v1[0] * v2[2];
   v_dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
}
float Util::mag(float* v) {
   return sqrt(dot(v, v));
}
void Util::scale(float* v_src, float s, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v_src[ndx] * s;
   }
}
void Util::scalei(float* v, float s) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v[ndx] *= s;
   }
}
void Util::scale(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] * v2[ndx];
   }
}
void Util::scalei(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] *= v2[ndx];
   }
}
void Util::norm(float* v_src, float* v_dest) {
   scale(v_src, 1.0f / mag(v_src), v_dest);
}
void Util::normi(float* v) {
   scalei(v, 1.0f / mag(v));
}
void Util::add(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] + v2[ndx];
   }
}
void Util::addi(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] += v2[ndx];
   }
}
void Util::sub(float* v1, float* v2, float* v_dest) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v_dest[ndx] = v1[ndx] - v2[ndx];
   }
}
void Util::subi(float* v1, float* v2) {
   for (int ndx = 0; ndx < 3; ++ndx) {
      v1[ndx] -= v2[ndx];
   }
}
void Util::print(float f, bool nl = false) {
   printf("%.2f", f);
   if (nl) {
      printf("\n");
   }
}
void Util::print(float* v, bool nl = false) {
   printf("[");
   print(v[0]);
   printf(",");
   print(v[1]);
   printf(",");
   print(v[2]);
   printf("]");
   if (nl) {
      printf("\n");
   }
}
