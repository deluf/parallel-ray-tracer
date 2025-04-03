#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include <stddef.h>

#include "vec.h"

typedef struct triangle_t {
    vec_t coords[3];
    vec_t ks;
    vec_t kd;
    vec_t kr;

    vec_t norm[2];
} triangle_t;

void triangle_init(triangle_t* t, const vec_t* a, const vec_t* b, const vec_t* c, const vec_t* ks, const vec_t* kd, const vec_t* kr);
triangle_t* triangles_load(const char* objname, const char* mtlname, size_t* size);

#endif