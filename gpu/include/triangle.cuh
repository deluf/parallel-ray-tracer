#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include <stddef.h>

#include "vec.cuh"

struct mat_t {
    vec_t ks;
    vec_t kd;
    vec_t kr;
};

struct triangle_t {
    vec_t coords[3];
    float centroid[3];
    int mat_idx;

    vec_t norm[2];
};

struct gpu_triangle_t {
    vec_t coords[3];
};

struct norm_t {
    vec_t norm[2];
};

void triangle_init(triangle_t* t, const vec_t* a, const vec_t* b, const vec_t* c, int mat_idx);
triangle_t* triangles_load(const char* objname, const char* mtlname, int* size, mat_t** mat);

#endif