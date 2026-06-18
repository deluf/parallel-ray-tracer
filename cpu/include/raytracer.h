#ifndef __RAYTRACER_H__
#define __RAYTRACER_H__

#include "vec.h"
#include "cam.h"
#include "triangle.h"

/**
 * Traces a ray and returns the color
 * 
 * @param origin The origin of the ray
 * @param dir The direction of the ray
 * @param iter The current bounce iteration
 * @return The computed color vector
 */
vec_t raytrace(vec_t origin, vec_t dir, int iter);

/**
 * Intersects a ray with a triangle
 * 
 * @param origin The origin of the ray
 * @param dir The direction of the ray
 * @param tr The triangle to test intersection with
 * @param norm_dir Pointer to store the normal direction flag
 * @return The intersection distance or FLT_MAX if no intersection
 */
float hit_triangle(const vec_t* origin, const vec_t* dir, const triangle_t* tr, int* norm_dir);

#endif
