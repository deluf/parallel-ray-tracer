#include "vec.cuh"

#include <cuda_runtime.h>  // norm3df(), rnorm3df()
#include <math.h>          // sqrtf(), fminf(), fmaxf()

/**
 * Computes the dot product of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The dot product
 */
__host__ __device__ float vec_dot(const vec_t* v1, const vec_t* v2)
{
    return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

/**
 * Computes the distance between two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The distance
 */
__host__ __device__ float vec_dist(const vec_t* v1, const vec_t* v2)
{
    float dx = v1->x - v2->x;
    float dy = v1->y - v2->y;
    float dz = v1->z - v2->z;
#ifdef __CUDA_ARCH__
    return norm3df(dx, dy, dz);
#else
    return sqrtf(dx * dx + dy * dy + dz * dz);
#endif
}

/**
 * Computes the magnitude of a vector
 * 
 * @param v1 Vector
 * @return The magnitude
 */
__host__ __device__ float vec_mag(const vec_t* v1)
{
#ifdef __CUDA_ARCH__
    return norm3df(v1->x, v1->y, v1->z);
#else
    return sqrtf(v1->x * v1->x + v1->y * v1->y + v1->z * v1->z);
#endif
}

/**
 * Computes the squared magnitude of a vector
 * 
 * @param v1 Vector
 * @return The squared magnitude
 */
__host__ __device__ float vec_mag2(const vec_t* v1)
{
    return v1->x * v1->x + v1->y * v1->y + v1->z * v1->z;
}

/**
 * Normalizes a vector in place
 * 
 * @param v1 Vector to normalize
 */
__host__ __device__ void vec_normalize(vec_t* v1)
{
#ifdef __CUDA_ARCH__
    *v1 = vec_mul(v1, rnorm3df(v1->x, v1->y, v1->z));
#else
    *v1 = vec_div(v1, vec_mag(v1));
#endif
}

/**
 * Multiplies a vector by a scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_mul(const vec_t* v1, float val)
{
    return vec_t
    {
        v1->x * val,
        v1->y * val,
        v1->z * val,
        0.0f
    };
}

/**
 * Component-wise multiplies two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_mul(const vec_t* v1, const vec_t* v2)
{
    return vec_t
    {
        v1->x * v2->x,
        v1->y * v2->y,
        v1->z * v2->z,
        0.0f
    };
}

/**
 * Adds two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_add(const vec_t* v1, const vec_t* v2)
{
    return vec_t
    {
        v1->x + v2->x,
        v1->y + v2->y,
        v1->z + v2->z,
        0.0f
    };
}

/**
 * Subtracts two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_sub(const vec_t* v1, const vec_t* v2)
{
    return vec_t
    {
        v1->x - v2->x,
        v1->y - v2->y,
        v1->z - v2->z,
        0.0f
    };
}

/**
 * Divides a vector by a scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_div(const vec_t* v1, float val)
{
    return vec_t
    {
        v1->x / val,
        v1->y / val,
        v1->z / val,
        0.0f
    };
}

/**
 * Computes the cross product of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_cross(const vec_t* v1, const vec_t* v2)
{
#ifdef __CUDA_ARCH__
    return vec_t
    {
        __fmaf_rn(v1->y, v2->z, -v1->z * v2->y),
        __fmaf_rn(v1->z, v2->x, -v1->x * v2->z),
        __fmaf_rn(v1->x, v2->y, -v1->y * v2->x),
        0.0f
    };
#else
    return vec_t
    {
        v1->y * v2->z - v1->z * v2->y,
        v1->z * v2->x - v1->x * v2->z,
        v1->x * v2->y - v1->y * v2->x,
        0.0f
    };
#endif
}

/**
 * Constrains a vector within a minimum and maximum vector bounds
 * 
 * @param v Vector to constrain
 * @param min Minimum bounds
 * @param max Maximum bounds
 */
__host__ __device__ void vec_constrain(vec_t* v, const vec_t* min, const vec_t* max)
{
    v->x = fmaxf(v->x, min->x);
    v->y = fmaxf(v->y, min->y);
    v->z = fmaxf(v->z, min->z);
    v->x = fminf(v->x, max->x);
    v->y = fminf(v->y, max->y);
    v->z = fminf(v->z, max->z);
}

/**
 * Returns a vector containing the minimum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The minimum vector
 */
__host__ __device__ vec_t vec_min(const vec_t* v1, const vec_t* v2)
{
    return vec_t
    {
        fminf(v1->x, v2->x),
        fminf(v1->y, v2->y),
        fminf(v1->z, v2->z),
        0.0f
    };
}

/**
 * Returns a vector containing the maximum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The maximum vector
 */
__host__ __device__ vec_t vec_max(const vec_t* v1, const vec_t* v2)
{
    return vec_t
    {
        fmaxf(v1->x, v2->x),
        fmaxf(v1->y, v2->y),
        fmaxf(v1->z, v2->z),
        0.0f
    };
}

/**
 * Computes a fused multiply-add of two vectors and a third vector component-wise
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @param v3 Third vector
 * @return The resulting vector
 */
__device__ vec_t vec_ma(const vec_t* v1, const vec_t* v2, const vec_t* v3)
{
    return vec_t
    {
        __fmaf_rn(v1->x, v2->x, v3->x),
        __fmaf_rn(v1->y, v2->y, v3->y),
        __fmaf_rn(v1->z, v2->z, v3->z),
        0.0f
    };
}

/**
 * Computes a fused multiply-add of a vector, a scalar, and a third vector
 * 
 * @param v1 First vector
 * @param t Scalar value
 * @param v3 Third vector
 * @return The resulting vector
 */
__device__ vec_t vec_ma(const vec_t* v1, float t, const vec_t* v3)
{
    return vec_t
    {
        __fmaf_rn(v1->x, t, v3->x),
        __fmaf_rn(v1->y, t, v3->y),
        __fmaf_rn(v1->z, t, v3->z),
        0.0f
    };
}

/**
 * Computes the dot product of two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The dot product
 */
__device__ __half hvec_dot(const hvec_t* v1, const hvec_t* v2)
{
    return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

/**
 * Computes the distance between two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The distance
 */
__device__ __half hvec_dist(const hvec_t* v1, const hvec_t* v2)
{
    __half dx = v1->x - v2->x;
    __half dy = v1->y - v2->y;
    __half dz = v1->z - v2->z;
    return hsqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Computes the magnitude of a half-precision vector
 * 
 * @param v1 Vector
 * @return The magnitude
 */
__device__ __half hvec_mag(const hvec_t* v1)
{
    return hsqrt(v1->x * v1->x + v1->y * v1->y + v1->z * v1->z);
}

/**
 * Normalizes a half-precision vector in place
 * 
 * @param v1 Vector to normalize
 */
__device__ void hvec_normalize(hvec_t* v1)
{
    *v1 = hvec_div(v1, hvec_mag(v1));
}

/**
 * Multiplies a half-precision vector by a half-precision scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
__device__ hvec_t hvec_mul(const hvec_t* v1, __half val)
{
    return hvec_t
    {
        v1->x * val,
        v1->y * val,
        v1->z * val
    };
}

/**
 * Adds two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__device__ hvec_t hvec_add(const hvec_t* v1, const hvec_t* v2)
{
    return hvec_t
    {
        v1->x + v2->x,
        v1->y + v2->y,
        v1->z + v2->z
    };
}

/**
 * Subtracts two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__device__ hvec_t hvec_sub(const hvec_t* v1, const hvec_t* v2)
{
    return hvec_t
    {
        v1->x - v2->x,
        v1->y - v2->y,
        v1->z - v2->z
    };
}

/**
 * Divides a half-precision vector by a half-precision scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
__device__ hvec_t hvec_div(const hvec_t* v1, __half val)
{
    return hvec_t
    {
        v1->x / val,
        v1->y / val,
        v1->z / val
    };
}

/**
 * Computes the cross product of two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__device__ hvec_t hvec_cross(const hvec_t* v1, const hvec_t* v2)
{
    return hvec_t
    {
        v1->y * v2->z - v1->z * v2->y,
        v1->z * v2->x - v1->x * v2->z,
        v1->x * v2->y - v1->y * v2->x
    };
}

/**
 * Constrains a half-precision vector within a minimum and maximum vector bounds
 * 
 * @param v Vector to constrain
 * @param min Minimum bounds
 * @param max Maximum bounds
 */
__device__ void hvec_constrain(hvec_t* v, const hvec_t* min, const hvec_t* max)
{
    v->x = __hmax(v->x, min->x);
    v->y = __hmax(v->y, min->y);
    v->z = __hmax(v->z, min->z);
    v->x = __hmin(v->x, max->x);
    v->y = __hmin(v->y, max->y);
    v->z = __hmin(v->z, max->z);
}

/**
 * Returns a half-precision vector containing the minimum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The minimum vector
 */
__device__ hvec_t hvec_min(const hvec_t* v1, const hvec_t* v2)
{
    return hvec_t
    {
        __hmin(v1->x, v2->x),
        __hmin(v1->y, v2->y),
        __hmin(v1->z, v2->z)
    };
}

/**
 * Returns a half-precision vector containing the maximum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The maximum vector
 */
__device__ hvec_t hvec_max(const hvec_t* v1, const hvec_t* v2)
{
    return hvec_t
    {
        __hmax(v1->x, v2->x),
        __hmax(v1->y, v2->y),
        __hmax(v1->z, v2->z)
    };
}
