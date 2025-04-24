#include "raytracer.cuh"
#include "gpu.cuh"
#include "options.cuh"

#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

#define EPSILON (1e-3)

__device__ static vec_t lambert_blinn(const vec_t* ks, const vec_t* kd, const vec_t* n, const vec_t* l, const vec_t* v, float dot){
    vec_t h = vec_add(l, v);
    vec_normalize(&h);
    
    float coeff = fmax(0.0f, vec_dot(n, &h));

    vec_t out;
    out = vec_mul(kd, fmaxf(0.0f, dot));
    out = vec_ma(ks, coeff, &out);
    return out;
}

__device__ float hit_triangle(const vec_t* origin, const vec_t* dir, const gpu_triangle_t* tr, int* norm_dir) {
    vec_t e1 = vec_sub(&tr->coords[1], &tr->coords[0]);
    vec_t e2 = vec_sub(&tr->coords[2], &tr->coords[0]);
    vec_t n = vec_cross(&e1, &e2);
    float det = -vec_dot(dir, &n);

    *norm_dir = det < 0.0f;

    float abs_det = fabsf(det);
    if(abs_det < EPSILON)
        return FLT_MAX;

    float invdet = 1.0f / det;
    vec_t ao = vec_sub(origin, &tr->coords[0]);
    vec_t dao = vec_cross(&ao, dir);

    float u = vec_dot(&e2, &dao) * invdet;
    float v = -vec_dot(&e1, &dao) * invdet;
    float t = vec_dot(&ao, &n) * invdet;

    if(t > EPSILON && u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f)
        return t;

    return FLT_MAX;
}

// light visibility - check if the ligh reach the points or is blocked by a sphere or traingle
__device__ static float light_v(const vec_t* origin, const vec_t* dir, const vec_t* n, const vec_t* light){
    vec_t tmp = vec_sub(origin, light);
    vec_t tmp2 = vec_sub(light, origin);
    float light_dist2 = vec_dot(&tmp, &tmp);
    if(vec_dot(&tmp2, n) < 0.0f)
        return 0.0f;
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
            final_col = vec_ma(&multiplier, &gpu_amb_light, &final_col);
            break;
        } 

        vec_t intersection = vec_ma(&dir, t, &origin);
        const mat_t* __restrict__ mat = &gpu_mats[gpu_mat_idx[index]];
        vec_t ks = mat->ks;
        vec_t kd = mat->kd;
        vec_t kr = mat->kr;
        vec_t n = gpu_norms[index].norm[norm_dir];
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
            float V = light_v(&intersection, &l, &n, &gpu_lights[i].pos);
            vec_t light_shade = vec_mul(&gpu_lights[i].kl, &col_ray);
            light_shade = vec_div(&light_shade, mag);
            col = vec_ma(&light_shade, V, &col);
        }

        final_col = vec_ma(&multiplier, &col, &final_col);

        if(vec_mag2(&multiplier) < EPSILON*EPSILON)
            break;

        multiplier = vec_mul(&multiplier, &kr);

        //real raytracing EXTREMELY HEAVY
        dir = vec_mul(&dir, -1.0f);
        dir = vec_ma(&n, 2.0f*fabsf(vec_dot(&dir, &n)), &dir);
        vec_normalize(&dir);
        origin = intersection;
    }

    return final_col;
}