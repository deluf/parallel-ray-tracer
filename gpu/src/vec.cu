#include "vec.cuh"

#include <cuda_runtime.h>

__host__ __device__ float vec_dot(const vec_t* v1, const vec_t* v2){
    return v1->x*v2->x + v1->y*v2->y + v1->z*v2->z;
}

__host__ __device__ float vec_dist(const vec_t* v1, const vec_t* v2){
    float dx = v1->x - v2->x;
    float dy = v1->y - v2->y;
    float dz = v1->z - v2->z;
    #ifdef __CUDA_ARCH__
    return norm3df(dx, dy, dz);
    #else
    return sqrtf(dx * dx + dy * dy + dz * dz);
    #endif
}

__host__ __device__ float vec_mag(const vec_t* v1){
    #ifdef __CUDA_ARCH__
    return norm3df(v1->x, v1->y, v1->z);
    #else
    return sqrtf(v1->x*v1->x + v1->y*v1->y + v1->z*v1->z);
    #endif
}

__host__ __device__ float vec_mag2(const vec_t* v1){
    return v1->x*v1->x + v1->y*v1->y + v1->z*v1->z;
}

__host__ __device__ void vec_normalize(vec_t* v1){
    #ifdef __CUDA_ARCH__
    *v1 = vec_mul(v1, rnorm3df(v1->x, v1->y, v1->z));
    #else
    *v1 = vec_div(v1, vec_mag(v1));
    #endif
}

__host__ __device__ vec_t vec_mul(const vec_t* v1, float val){
    return vec_t{v1->x*val, v1->y*val, v1->z*val};
}

__host__ __device__ vec_t vec_mul(const vec_t* v1, const vec_t* v2){
    return vec_t{v1->x*v2->x, v1->y*v2->y, v1->z*v2->z};
}

__host__ __device__ vec_t vec_add(const vec_t* v1, const vec_t* v2){
    return vec_t{v1->x+v2->x, v1->y+v2->y, v1->z+v2->z};
}

__host__ __device__ vec_t vec_sub(const vec_t* v1, const vec_t* v2){
    return vec_t{v1->x-v2->x, v1->y-v2->y, v1->z-v2->z};
}

__host__ __device__ vec_t vec_div(const vec_t* v1, float val){
    return vec_t{v1->x/val, v1->y/val, v1->z/val};
}

__host__ __device__ vec_t vec_cross(const vec_t* v1, const vec_t* v2){
    #ifdef __CUDA_ARCH__
    return vec_t{
        __fmaf_rn(v1->y, v2->z, -v1->z*v2->y),
        __fmaf_rn(v1->z, v2->x, -v1->x*v2->z),
        __fmaf_rn(v1->x, v2->y, -v1->y*v2->x)
    };
    #else
    return vec_t{
        v1->y*v2->z - v1->z*v2->y,
        v1->z*v2->x - v1->x*v2->z,
        v1->x*v2->y - v1->y*v2->x
    };
    #endif
}

__host__ __device__ void vec_constrain(vec_t* v, const vec_t* min, const vec_t* max){
    v->x = fmaxf(v->x, min->x);
    v->y = fmaxf(v->y, min->y);
    v->z = fmaxf(v->z, min->z);
    v->x = fminf(v->x, max->x);
    v->y = fminf(v->y, max->y);
    v->z = fminf(v->z, max->z);
}

__host__ __device__ vec_t vec_min(const vec_t* v1, const vec_t* v2){
    return vec_t{
        fminf(v1->x, v2->x),
        fminf(v1->y, v2->y),
        fminf(v1->z, v2->z)
    };
}
__host__ __device__ vec_t vec_max(const vec_t* v1, const vec_t* v2){
    return vec_t{
        fmaxf(v1->x, v2->x),
        fmaxf(v1->y, v2->y),
        fmaxf(v1->z, v2->z)
    };
}

__device__ vec_t vec_ma(const vec_t* v1, const vec_t* v2, const vec_t* v3){
    return vec_t{
        __fmaf_rn(v1->x, v2->x, v3->x),
        __fmaf_rn(v1->y, v2->y, v3->y),
        __fmaf_rn(v1->z, v2->z, v3->z)
    };
}

__device__ vec_t vec_ma(const vec_t* v1, float t, const vec_t* v3){
    return vec_t{
        __fmaf_rn(v1->x, t, v3->x),
        __fmaf_rn(v1->y, t, v3->y),
        __fmaf_rn(v1->z, t, v3->z)
    };
}

/// __half VEC

__device__ __half hvec_dot(const hvec_t* v1, const hvec_t* v2){
    return v1->x*v2->x + v1->y*v2->y + v1->z*v2->z;
}

__device__ __half hvec_dist(const hvec_t* v1, const hvec_t* v2){
    __half dx = v1->x - v2->x;
    __half dy = v1->y - v2->y;
    __half dz = v1->z - v2->z;
    return hsqrt(dx*dx + dy*dy + dz*dz);
}

__device__ __half hvec_mag(const hvec_t* v1){
    return hsqrt(v1->x*v1->x + v1->y*v1->y + v1->z*v1->z);
}

__device__ void hvec_normalize(hvec_t* v1){
    *v1 = hvec_div(v1, hvec_mag(v1));
}

__device__ hvec_t hvec_mul(const hvec_t* v1, __half val){
    return hvec_t{v1->x*val, v1->y*val, v1->z*val};
}

__device__ hvec_t hvec_add(const hvec_t* v1, const hvec_t* v2){
    return hvec_t{v1->x+v2->x, v1->y+v2->y, v1->z+v2->z};
}

__device__ hvec_t hvec_sub(const hvec_t* v1, const hvec_t* v2){
    return hvec_t{v1->x-v2->x, v1->y-v2->y, v1->z-v2->z};
}

__device__ hvec_t hvec_div(const hvec_t* v1, __half val){
    return hvec_t{v1->x/val, v1->y/val, v1->z/val};
}

__device__ hvec_t hvec_cross(const hvec_t* v1, const hvec_t* v2){
    return hvec_t{
        v1->y*v2->z - v1->z*v2->y,
        v1->z*v2->x - v1->x*v2->z,
        v1->x*v2->y - v1->y*v2->x
    };
}

__device__ void hvec_constrain(hvec_t* v, const hvec_t* min, const hvec_t* max){
    v->x = __hmax(v->x, min->x);
    v->y = __hmax(v->y, min->y);
    v->z = __hmax(v->z, min->z);
    v->x = __hmin(v->x, max->x);
    v->y = __hmin(v->y, max->y);
    v->z = __hmin(v->z, max->z);
}

__device__ hvec_t hvec_min(const hvec_t* v1, const hvec_t* v2){
    return hvec_t{
        __hmin(v1->x, v2->x),
        __hmin(v1->y, v2->y),
        __hmin(v1->z, v2->z)
    };
}
__device__ hvec_t hvec_max(const hvec_t* v1, const hvec_t* v2){
    return hvec_t{
        __hmax(v1->x, v2->x),
        __hmax(v1->y, v2->y),
        __hmax(v1->z, v2->z)
    };
}

