#ifndef __TRIANGLE_CUH__
#define __TRIANGLE_CUH__

#include <stddef.h>  // size_t

#include "vec.cuh"

typedef struct mat_t
{
    vec_t ks;
    vec_t kd;
    vec_t kr;
}
mat_t;

typedef struct triangle_t
{
    vec_t coords[3];
    float centroid[3];
    int mat_idx;
    vec_t norm[2];
}
triangle_t;

typedef struct gpu_triangle_t
{
    vec_t coords[3];
}
gpu_triangle_t;

typedef struct norm_t
{
    vec_t norm[2] ;
}
norm_t;

/**
 * Initializes a triangle structure
 *
 * @param t Pointer to the triangle to initialize
 * @param a First vertex vector coordinate
 * @param b Second vertex vector coordinate
 * @param c Third vertex vector coordinate
 * @param mat_idx Material index for shading
 */
void triangle_init(triangle_t* t, const vec_t* a, const vec_t* b, const vec_t* c, int mat_idx);

/**
 * Loads triangles and materials from OBJ and MTL files
 *
 * @param objname Path to the OBJ file
 * @param mtlname Path to the MTL file
 * @param size Pointer to store the number of loaded triangles
 * @param mats Pointer to store the allocated material array
 * @return Pointer to dynamically allocated array of triangles, or NULL on failure
 */
triangle_t* triangles_load(const char* objname, const char* mtlname, size_t* size, mat_t** mats);

#endif
