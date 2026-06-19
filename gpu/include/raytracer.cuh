#ifndef __RAYTRACER_CUH__
#define __RAYTRACER_CUH__

#include "vec.cuh"
#include "cam.cuh"
#include "triangle.cuh"

/**
 * Traces a ray and returns the final color vector
 * 
 * @param origin The origin vector of the ray
 * @param dir The direction vector of the ray
 * @return Computed color vector
 */
__device__ vec_t raytrace(vec_t origin, vec_t dir);

/**
 * Intersects a ray with a triangle on the device
 * 
 * @param origin The origin vector of the ray
 * @param dir The direction vector of the ray
 * @param tr Pointer to the GPU triangle structure
 * @param norm_dir Pointer to store the normal direction index
 * @return Intersection distance, or FLT_MAX if no intersection
 */
__device__ float hit_triangle(const vec_t* origin, const vec_t* dir, const gpu_triangle_t* tr, int* norm_dir);

#endif
