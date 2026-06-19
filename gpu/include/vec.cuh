#ifndef __VEC_CUH__
#define __VEC_CUH__

#include <cuda_fp16.h>  // __half, __half2, half, half2

typedef struct vec_t
{
    union
    {
        float arr[4];
        float4 fl4;
        struct
        {
            float2 xy;
            float2 zw;
        };
        struct
        {
            float x;
            float y;
            float z;
            float w;
        };
        struct
        {
            float r;
            float g;
            float b;
            float a;
        };
    };
}
vec_t;

typedef struct hvec_t
{
    union
    {
        __half arr[4];
        struct
        {
            __half2 xy;
            __half2 zw;
        };
        struct
        {
            __half x;
            __half y;
            __half z;
            __half w;
        };
        struct
        {
            __half r;
            __half g;
            __half b;
            __half a;
        };
    };

    __host__ __device__ hvec_t(__half x, __half y, __half z)
        : x(x), y(y), z(z), w(0.0f)
    {
    }
    
    __device__ hvec_t()
    {
        zw = xy = make_half2(0.0f, 0.0f);
    }
    
    __host__ __device__ hvec_t(const hvec_t& other)
        : xy(other.xy), zw(other.zw)
    {
    }
    
    __host__ __device__ hvec_t& operator=(const hvec_t& other)
    {
        xy = other.xy;
        zw = other.zw;
        return *this;
    }
}
hvec_t;

/**
 * Computes the dot product of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The dot product
 */
__host__ __device__ float vec_dot(const vec_t* v1, const vec_t* v2);

/**
 * Computes the distance between two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The distance
 */
__host__ __device__ float vec_dist(const vec_t* v1, const vec_t* v2);

/**
 * Computes the magnitude of a vector
 * 
 * @param v1 Vector
 * @return The magnitude
 */
__host__ __device__ float vec_mag(const vec_t* v1);

/**
 * Computes the squared magnitude of a vector
 * 
 * @param v1 Vector
 * @return The squared magnitude
 */
__host__ __device__ float vec_mag2(const vec_t* v1);

/**
 * Normalizes a vector in place
 * 
 * @param v1 Vector to normalize
 */
__host__ __device__ void vec_normalize(vec_t* v1);

/**
 * Multiplies a vector by a scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_mul(const vec_t* v1, float val);

/**
 * Component-wise multiplies two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_mul(const vec_t* v1, const vec_t* v2);

/**
 * Adds two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_add(const vec_t* v1, const vec_t* v2);

/**
 * Subtracts two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_sub(const vec_t* v1, const vec_t* v2);

/**
 * Divides a vector by a scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_div(const vec_t* v1, float val);

/**
 * Computes the cross product of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__host__ __device__ vec_t vec_cross(const vec_t* v1, const vec_t* v2);

/**
 * Constrains a vector within a minimum and maximum vector bounds
 * 
 * @param v Vector to constrain
 * @param min Minimum bounds
 * @param max Maximum bounds
 */
__host__ __device__ void vec_constrain(vec_t* v, const vec_t* min, const vec_t* max);

/**
 * Returns a vector containing the minimum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The minimum vector
 */
__host__ __device__ vec_t vec_min(const vec_t* v1, const vec_t* v2);

/**
 * Returns a vector containing the maximum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The maximum vector
 */
__host__ __device__ vec_t vec_max(const vec_t* v1, const vec_t* v2);

/**
 * Computes a fused multiply-add of two vectors and a third vector component-wise
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @param v3 Third vector
 * @return The resulting vector
 */
__device__ vec_t vec_ma(const vec_t* v1, const vec_t* v2, const vec_t* v3);

/**
 * Computes a fused multiply-add of a vector, a scalar, and a third vector
 * 
 * @param v1 First vector
 * @param t Scalar value
 * @param v3 Third vector
 * @return The resulting vector
 */
__device__ vec_t vec_ma(const vec_t* v1, float t, const vec_t* v3);

/**
 * Computes the dot product of two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The dot product
 */
__device__ __half hvec_dot(const hvec_t* v1, const hvec_t* v2);

/**
 * Computes the distance between two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The distance
 */
__device__ __half hvec_dist(const hvec_t* v1, const hvec_t* v2);

/**
 * Computes the magnitude of a half-precision vector
 * 
 * @param v1 Vector
 * @return The magnitude
 */
__device__ __half hvec_mag(const hvec_t* v1);

/**
 * Normalizes a half-precision vector in place
 * 
 * @param v1 Vector to normalize
 */
__device__ void hvec_normalize(hvec_t* v1);

/**
 * Multiplies a half-precision vector by a half-precision scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
__device__ hvec_t hvec_mul(const hvec_t* v1, __half val);

/**
 * Adds two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__device__ hvec_t hvec_add(const hvec_t* v1, const hvec_t* v2);

/**
 * Subtracts two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__device__ hvec_t hvec_sub(const hvec_t* v1, const hvec_t* v2);

/**
 * Divides a half-precision vector by a half-precision scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
__device__ hvec_t hvec_div(const hvec_t* v1, __half val);

/**
 * Computes the cross product of two half-precision vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
__device__ hvec_t hvec_cross(const hvec_t* v1, const hvec_t* v2);

/**
 * Constrains a half-precision vector within a minimum and maximum vector bounds
 * 
 * @param v Vector to constrain
 * @param min Minimum bounds
 * @param max Maximum bounds
 */
__device__ void hvec_constrain(hvec_t* v, const hvec_t* min, const hvec_t* max);

/**
 * Returns a half-precision vector containing the minimum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The minimum vector
 */
__device__ hvec_t hvec_min(const hvec_t* v1, const hvec_t* v2);

/**
 * Returns a half-precision vector containing the maximum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The maximum vector
 */
__device__ hvec_t hvec_max(const hvec_t* v1, const hvec_t* v2);

#endif
