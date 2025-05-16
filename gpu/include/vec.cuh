#ifndef __VEC_H__
#define __VEC_H__

#include <cuda_fp16.h>

struct vec_t {
    union {
        float arr[4];
        float4 fl4;
        struct {
            float2 xy;
            float2 zw;
        };
        struct {
            float x;
            float y;
            float z;
            float w;
        };
        struct {
            float r;
            float g;
            float b;
            float a;
        };
    };
};


struct hvec_t {
    union {
        __half arr[4];
        struct {
            __half2 xy;
            __half2 zw;
        };
        struct {
            __half x;
            __half y;
            __half z;
            __half w;
        };
        struct {
            __half r;
            __half g;
            __half b;
            __half a;
        };
    };

    __host__ __device__ hvec_t(__half x, __half y, __half z) : x(x), y(y), z(z), w(0.0f) {}
    __device__ hvec_t() { zw = xy = make_half2(0, 0); }
    __host__ __device__ hvec_t(const hvec_t& other) : xy(other.xy), zw(other.zw) {}
    __host__ __device__ hvec_t& operator=(const hvec_t& other) { xy = other.xy; zw = other.zw; return *this; }
    
};

__host__ __device__ float vec_dot(const vec_t* v1, const vec_t* v2);
__host__ __device__ float vec_dist(const vec_t* v1, const vec_t* v2);
__host__ __device__ float vec_mag(const vec_t* v1);
__host__ __device__ float vec_mag2(const vec_t* v1);
__host__ __device__ void vec_normalize(vec_t* v1);
__host__ __device__ vec_t vec_mul(const vec_t* v1, float val);
__host__ __device__ vec_t vec_mul(const vec_t* v1, const vec_t* v2);
__host__ __device__ vec_t vec_add(const vec_t* v1, const vec_t* v2);
__host__ __device__ vec_t vec_sub(const vec_t* v1, const vec_t* v2);
__host__ __device__ vec_t vec_div(const vec_t* v1, float val);
__host__ __device__ vec_t vec_cross(const vec_t* v1, const vec_t* v2);
__host__ __device__ void vec_constrain(vec_t* v, const vec_t* min, const vec_t* max);
__host__ __device__ vec_t vec_min(const vec_t* v1, const vec_t* v2);
__host__ __device__ vec_t vec_max(const vec_t* v1, const vec_t* v2);
__device__ vec_t vec_ma(const vec_t* v1, const vec_t* v2, const vec_t* v3);
__device__ vec_t vec_ma(const vec_t* v1, float t, const vec_t* v3);


__device__ __half hvec_dot(const hvec_t* v1, const hvec_t* v2);
__device__ __half hvec_dist(const hvec_t* v1, const hvec_t* v2);
__device__ __half hvec_mag(const hvec_t* v1);
__device__ void hvec_normalize(hvec_t* v1);
__device__ hvec_t hvec_mul(const hvec_t* v1, __half val);
__device__ hvec_t hvec_add(const hvec_t* v1, const hvec_t* v2);
__device__ hvec_t hvec_sub(const hvec_t* v1, const hvec_t* v2);
__device__ hvec_t hvec_div(const hvec_t* v1, __half val);
__device__ hvec_t hvec_cross(const hvec_t* v1, const hvec_t* v2);
__device__ void hvec_constrain(hvec_t* v, const hvec_t* min, const hvec_t* max);
__device__ hvec_t hvec_min(const hvec_t* v1, const hvec_t* v2);
__device__ hvec_t hvec_max(const hvec_t* v1, const hvec_t* v2);

#endif