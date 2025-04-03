#ifndef __SPHERE_H__
#define __SPHERE_H__

#include <stddef.h>

#include "vec.h"

typedef struct sphere_t {
    vec_t pos;
    float r;
    vec_t ks;
    vec_t kd;
    vec_t kr;
} sphere_t;

void sphere_init(sphere_t* sp, float r, const vec_t* pos, const vec_t* kd, const vec_t* ks);
sphere_t* sphere_load(const char* filename, size_t* size);

#endif