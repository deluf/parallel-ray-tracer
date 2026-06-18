#ifndef __VEC_H__
#define __VEC_H__

/**
 * 3D vector struct
 */
typedef struct vec_t 
{
    union 
    {
        float arr[3];
        struct 
        {
            float x;
            float y;
            float z;
        };
        struct 
        {
            float r;
            float g;
            float b;
        };
    };
} 
vec_t;

/**
 * Computes the dot product of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The dot product
 */
float vec_dot(const vec_t* v1, const vec_t* v2);

/**
 * Computes the distance between two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The distance
 */
float vec_dist(const vec_t* v1, const vec_t* v2);

/**
 * Computes the magnitude of a vector
 * 
 * @param v1 Vector
 * @return The magnitude
 */
float vec_mag(const vec_t* v1);

/**
 * Normalizes a vector in place
 * 
 * @param v1 Vector to normalize
 */
void vec_normalize(vec_t* v1);

/**
 * Multiplies a vector by a scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
vec_t vec_mul(const vec_t* v1, float val);

/**
 * Adds two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
vec_t vec_add(const vec_t* v1, const vec_t* v2);

/**
 * Subtracts two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
vec_t vec_sub(const vec_t* v1, const vec_t* v2);

/**
 * Divides a vector by a scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
vec_t vec_div(const vec_t* v1, float val);

/**
 * Computes the cross product of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
vec_t vec_cross(const vec_t* v1, const vec_t* v2);

/**
 * Constrains a vector within a minimum and maximum vector bounds
 * 
 * @param v Vector to constrain
 * @param min Minimum bounds
 * @param max Maximum bounds
 */
void vec_constrain(vec_t* v, const vec_t* min, const vec_t* max);

/**
 * Returns a vector containing the minimum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The minimum vector
 */
vec_t vec_min(const vec_t* v1, const vec_t* v2);

/**
 * Returns a vector containing the maximum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The maximum vector
 */
vec_t vec_max(const vec_t* v1, const vec_t* v2);

#endif
