#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include <stddef.h>  // size_t

#include "vec.h"

/**
 * Triangle structure
 */
typedef struct triangle_t
{
    vec_t coords[3];
    float centroid[3];
    vec_t ks;
    vec_t kd;
    vec_t kr;
    vec_t norm[2];
} triangle_t;

/**
 * Initializes a triangle object
 *
 * @param t Pointer to the triangle
 * @param a Pointer to the first vertex
 * @param b Pointer to the second vertex
 * @param c Pointer to the third vertex
 * @param ks Pointer to the specular vector
 * @param kd Pointer to the diffuse vector
 * @param kr Pointer to the reflection vector
 */
void triangle_init(triangle_t* t, const vec_t* a, const vec_t* b, const vec_t* c, const vec_t* ks, const vec_t* kd, const vec_t* kr);

/**
 * Loads triangles from a wavefront obj file and an mtl file
 *
 * @param objname Path to the obj file
 * @param mtlname Path to the mtl file
 * @param size Pointer to store the number of loaded triangles
 * @return Pointer to the array of loaded triangles
 */
triangle_t* triangles_load(const char* objname, const char* mtlname, size_t* size);

#endif
