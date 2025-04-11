#ifndef __RAYTRACER_H__
#define __RAYTRACER_H__

#include "vec.h"
#include "cam.h"
#include "triangle.h"

vec_t raytrace(vec_t origin, vec_t dir, int iter);

float hit_triangle(const vec_t* origin, const vec_t* dir, const triangle_t* tr, int* norm_dir);

#endif