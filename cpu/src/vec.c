#include "vec.h"

#include <math.h>  // sqrtf(), fminf(), fmaxf()

/**
 * Computes the dot product of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The dot product
 */
float vec_dot(const vec_t* v1, const vec_t* v2)
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
float vec_dist(const vec_t* v1, const vec_t* v2)
{
    float dx = v1->x - v2->x;
    float dy = v1->y - v2->y;
    float dz = v1->z - v2->z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

/**
 * Computes the magnitude of a vector
 * 
 * @param v1 Vector
 * @return The magnitude
 */
float vec_mag(const vec_t* v1)
{
    return sqrtf(v1->x * v1->x + v1->y * v1->y + v1->z * v1->z);
}

/**
 * Normalizes a vector in place
 * 
 * @param v1 Vector to normalize
 */
void vec_normalize(vec_t* v1)
{
    *v1 = vec_div(v1, vec_mag(v1));
}

/**
 * Multiplies a vector by a scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
vec_t vec_mul(const vec_t* v1, float val)
{
    return (vec_t){v1->x * val, v1->y * val, v1->z * val};
}

/**
 * Adds two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
vec_t vec_add(const vec_t* v1, const vec_t* v2)
{
    return (vec_t){v1->x + v2->x, v1->y + v2->y, v1->z + v2->z};
}

/**
 * Subtracts two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
vec_t vec_sub(const vec_t* v1, const vec_t* v2)
{
    return (vec_t){v1->x - v2->x, v1->y - v2->y, v1->z - v2->z};
}

/**
 * Divides a vector by a scalar
 * 
 * @param v1 Vector
 * @param val Scalar value
 * @return The resulting vector
 */
vec_t vec_div(const vec_t* v1, float val)
{
    return (vec_t){v1->x / val, v1->y / val, v1->z / val};
}

/**
 * Computes the cross product of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The resulting vector
 */
vec_t vec_cross(const vec_t* v1, const vec_t* v2)
{
    return (vec_t){
        v1->y * v2->z - v1->z * v2->y,
        v1->z * v2->x - v1->x * v2->z,
        v1->x * v2->y - v1->y * v2->x
    };
}

/**
 * Constrains a vector within a minimum and maximum vector bounds
 * 
 * @param v Vector to constrain
 * @param min Minimum bounds
 * @param max Maximum bounds
 */
void vec_constrain(vec_t* v, const vec_t* min, const vec_t* max)
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
vec_t vec_min(const vec_t* v1, const vec_t* v2)
{
    return (vec_t){
        fminf(v1->x, v2->x),
        fminf(v1->y, v2->y),
        fminf(v1->z, v2->z)
    };
}

/**
 * Returns a vector containing the maximum components of two vectors
 * 
 * @param v1 First vector
 * @param v2 Second vector
 * @return The maximum vector
 */
vec_t vec_max(const vec_t* v1, const vec_t* v2)
{
    return (vec_t){
        fmaxf(v1->x, v2->x),
        fmaxf(v1->y, v2->y),
        fmaxf(v1->z, v2->z)
    };
}
