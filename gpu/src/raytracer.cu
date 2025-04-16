#include "raytracer.cuh"
#include "gpu.cuh"
#include "options.cuh"

#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

extern int triangles_len;
extern triangle_t* triangles;

extern int lights_len;
extern light_t* lights;

extern vec_t amb_light;

#define EPSILON (1e-3)

__device__ static vec_t lambert_blinn(const vec_t* ks, const vec_t* kd, const vec_t* n, const vec_t* l, const vec_t* v, float dot){
    vec_t h = vec_add(l, v);
    vec_normalize(&h);
    
    float coeff = fmax(0.0f, vec_dot(n, &h));

    vec_t out;
    out.r = kd->r*fmaxf(0.0f, dot)+ks->r*coeff;
    out.g = kd->g*fmaxf(0.0f, dot)+ks->g*coeff;
    out.b = kd->b*fmaxf(0.0f, dot)+ks->b*coeff;

    return out;
}

__device__ float hit_triangle(const vec_t* origin, const vec_t* dir, const triangle_t* tr, int* norm_dir){
    *norm_dir = 0;
    vec_t e1 = vec_sub(&tr->coords[1], &tr->coords[0]);
    vec_t e2 = vec_sub(&tr->coords[2], &tr->coords[0]);
    vec_t n = vec_cross(&e1, &e2);
    volatile float det = -vec_dot(dir, &n);
    float invdet = 1.0/det;
    vec_t ao = vec_sub(origin, &tr->coords[0]);
    vec_t dao = vec_cross(&ao, dir);
    volatile float u = vec_dot(&e2, &dao)*invdet;
    volatile float v = -vec_dot(&e1, &dao)*invdet;
    volatile float t = vec_dot(&ao, &n)*invdet;
    if(det > 0 && t > EPSILON && u > 0 && v > 0 && (u+v) < 1){
        return t;  
    }

    *norm_dir = 1;
    e2 = vec_sub(&tr->coords[1], &tr->coords[0]);
    e1 = vec_sub(&tr->coords[2], &tr->coords[0]);
    n = vec_cross(&e1, &e2);
    det = -vec_dot(dir, &n);
    invdet = 1.0/det;
    ao = vec_sub(origin, &tr->coords[0]);
    dao = vec_cross(&ao, dir);
    u = vec_dot(&e2, &dao)*invdet;
    v = -vec_dot(&e1, &dao)*invdet;
    t = vec_dot(&ao, &n)*invdet;
    if(det > 0 && t > EPSILON && u > 0 && v > 0 && (u+v) < 1)
      return t;  
  
    return FLT_MAX;
}

// light visibility - check if the ligh reach the points or is blocked by a sphere or traingle
__device__ static int light_v(const vec_t* origin, const vec_t* dir, const vec_t* n, const vec_t* light){
    vec_t tmp = vec_sub(origin, light);
    vec_t tmp2 = vec_sub(light, origin);
    float light_dist2 = vec_dot(&tmp, &tmp);
    if(vec_dot(&tmp2, n) < 0)
        return 0;
    //check nearest triangle
    float t = FLT_MAX;
    return bvh_light_traverse(0, origin, dir, &t, light_dist2);
}

__device__ vec_t raytrace(vec_t origin, vec_t dir){
    vec_t final_col = {0.0f, 0.0f, 0.0f};
    vec_t multiplier = {1.0f, 1.0f, 1.0f};

    for(int iter = 0; iter < BOUNCES; iter++){
        vec_t col = {0.0f, 0.0f, 0.0f};
        int index = -1;
        float t = FLT_MAX;
        int norm_dir = 0;
        bvh_traverse(0, &origin, &dir, &norm_dir, &t, &index);
        if(index == -1){
            final_col.r += multiplier.r*gpu_amb_light.r;
            final_col.g += multiplier.g*gpu_amb_light.g;
            final_col.b += multiplier.b*gpu_amb_light.b;
            break;
        } 

        vec_t dir_scaled = vec_mul(&dir, t);
        vec_t intersection = vec_add(&origin, &dir_scaled);
        //if intersection is sphere
        vec_t ks;
        vec_t kd;
        vec_t kr;
        vec_t n;
        ks = gpu_triangles[index].ks;
        kd = gpu_triangles[index].kd;
        kr = gpu_triangles[index].kr;
        n = gpu_triangles[index].norm[norm_dir];
        //apply ambient light
        col.r = kd.r*gpu_amb_light.r;
        col.g = kd.g*gpu_amb_light.g;
        col.b = kd.b*gpu_amb_light.b;
        dir = vec_mul(&dir, -1.0f);
        //apply point lights
        for(int i = 0; i < gpu_lights_len; i++){
            vec_t l = vec_sub(&gpu_lights[i].pos, &intersection);
            float mag = vec_mag(&l);
            l = vec_div(&l, mag);
            mag *= mag;
            float n_dot_l = vec_dot(&n, &l);
            vec_t col_ray = lambert_blinn(&ks, &kd, &n, &l, &dir, n_dot_l);
            int V = light_v(&intersection, &l, &n, &gpu_lights[i].pos);
            col.r += V*gpu_lights[i].kl.r*col_ray.r/mag;
            col.g += V*gpu_lights[i].kl.g*col_ray.g/mag;
            col.b += V*gpu_lights[i].kl.b*col_ray.b/mag;
        }

        final_col.r += multiplier.r*col.r;
        final_col.g += multiplier.g*col.g;
        final_col.b += multiplier.b*col.b;

        multiplier.r *= kr.r;
        multiplier.g *= kr.g;
        multiplier.b *= kr.b;

        if(vec_mag(&multiplier) < EPSILON)
            break;

        //real raytracing EXTREMELY HEAVY
        dir = vec_mul(&dir, -1);
        vec_t n_scaled = vec_mul(&n, 2*fabsf(vec_dot(&dir, &n)));
        dir = vec_add(&dir, &n_scaled);
        vec_normalize(&dir);
        origin = intersection;
    }

    return final_col;
}