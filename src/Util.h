#ifndef _UTIL_H_
#define _UTIL_H_

namespace Util {
   void copy(float* v_src, float* v_dest);
   float dot(float* v1, float* v2);
   void cross(float* v1, float* v2, float* v_dest);
   float mag(float* v);
   void norm(float* v_src, float* v_dest);
   void normi(float* v);
   void add(float* v1, float* v2, float* v_dest);
   void addi(float* v1, float* v2);
   void sub(float* v1, float* v2, float* v_dest);
   void subi(float* v1, float* v2);
   void scale(float* v_src, float s, float* v_dest);
   void scalei(float* v, float s);
   void scale(float* v1, float* v2, float* v_dest);
   void scalei(float* v1, float* v2);
   void print(float f, bool nl);
   void print(float* v, bool nl);
}

#endif
