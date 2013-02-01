#ifndef _UTIL_H_
#define _UTIL_H_

#include <cuda.h>
#define CUDA_CALLABLE __host__ __device__

CUDA_CALLABLE float Q_rsqrt(float number);

CUDA_CALLABLE void zero_vector(float* v);
CUDA_CALLABLE void copy(float* v_src, float* v_dest);
CUDA_CALLABLE float dot(float* v1, float* v2);
CUDA_CALLABLE void cross(float* v1, float* v2, float* v_dest);
CUDA_CALLABLE float mag(float* v);
CUDA_CALLABLE void norm(float* v_src, float* v_dest);
CUDA_CALLABLE void norm_i(float* v);
CUDA_CALLABLE void add(float* v1, float* v2, float* v_dest);
CUDA_CALLABLE void add_i(float* v1, float* v2);
CUDA_CALLABLE void sub(float* v1, float* v2, float* v_dest);
CUDA_CALLABLE void sub_i(float* v1, float* v2);
CUDA_CALLABLE void scale(float* v_src, float s, float* v_dest);
CUDA_CALLABLE void scale_i(float* v, float s);
CUDA_CALLABLE void scalev(float* v1, float* v2, float* v_dest);
CUDA_CALLABLE void scalev_i(float* v1, float* v2);

#endif
