#ifndef __VEC_H__
#define __VEC_H__

typedef struct vec_t {
    union {
        float arr[4];
        float4 fl4;
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
} vec_t;

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

#endif