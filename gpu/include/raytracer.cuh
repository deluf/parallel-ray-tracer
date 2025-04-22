#ifndef __RAYTRACER_H__
#define __RAYTRACER_H__

#include "vec.cuh"
#include "cam.cuh"
#include "triangle.cuh"

__device__ vec_t raytrace(vec_t origin, vec_t dir);
__device__ float hit_triangle(const vec_t* origin, const vec_t* dir, const gpu_triangle_t* tr, int* norm_dir);

#endif