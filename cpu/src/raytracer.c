#include "raytracer.h"
#include "triangle.h"
#include "light.h"
#include "bvh.h"
#include "options.h"

#include <float.h>
#include <math.h>
#include <stdio.h>

extern size_t triangles_len;
extern triangle_t* triangles;

extern size_t lights_len;
extern light_t* lights;

extern vec_t amb_light;

const float EPSILON = 1e-3;

static vec_t lambert_blinn(const vec_t* ks, const vec_t* kd, const vec_t* n, const vec_t* l, const vec_t* v, float dot){
    vec_t h = vec_add(l, v);
    vec_normalize(&h);
    
    float coeff = fmax(0, vec_dot(n, &h));

    vec_t out;
    out.r = kd->r*fmaxf(0, dot)+ks->r*coeff;
    out.g = kd->g*fmaxf(0, dot)+ks->g*coeff;
    out.b = kd->b*fmaxf(0, dot)+ks->b*coeff;

    return out;
}

float hit_triangle(const vec_t* origin, const vec_t* dir, const triangle_t* tr, int* norm_dir){
    *norm_dir = 0;
    vec_t e1 = vec_sub(&tr->coords[1], &tr->coords[0]);
    vec_t e2 = vec_sub(&tr->coords[2], &tr->coords[0]);
    vec_t n = vec_cross(&e1, &e2);
    float det = -vec_dot(dir, &n);
    float invdet = 1.0/det;
    vec_t ao = vec_sub(origin, &tr->coords[0]);
    vec_t dao = vec_cross(&ao, dir);
    float u = vec_dot(&e2, &dao)*invdet;
    float v = -vec_dot(&e1, &dao)*invdet;
    float t = vec_dot(&ao, &n)*invdet;
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
static int light_v(const vec_t* origin, const vec_t* dir, const vec_t* n, const vec_t* light){
    vec_t tmp = vec_sub(origin, light);
    vec_t tmp2 = vec_sub(light, origin);
    float light_dist2 = vec_dot(&tmp, &tmp);
    if(vec_dot(&tmp2, n) < 0)
        return 0;
    //check nearest triangle
    int dummy;
    int index = -1;
    float t = FLT_MAX;
    #if USE_BVH == 1
        #if USE_BVH_FAST_LIGHT == 1
        return bvh_light_traverse(0, origin, dir, &t, light_dist2);
        #else    
        bvh_traverse(0, origin, dir, &dummy, &t, &index);
        if(index != -1){
            vec_t dir_scaled = vec_mul(dir, t);
            vec_t intersection = vec_add(origin, &dir_scaled);
            vec_t o_minus_i = vec_sub(origin, &intersection);
            if(light_dist2 > vec_dot(&o_minus_i, &o_minus_i) )
                return 0;
        }
        #endif
    #else
    for(int i = 0; i < triangles_len; i++){
        int dummy;
        float t = hit_triangle(origin, dir, &triangles[i], &dummy);
        if(t > EPSILON){
            vec_t dir_scaled = vec_mul(dir, t);
            vec_t intersection = vec_add(origin, &dir_scaled);
            vec_t o_minus_i = vec_sub(origin, &intersection);
            if(light_dist2 > vec_dot(&o_minus_i, &o_minus_i) )
                return 0;
        }
    }
    #endif
    return 1;
}

vec_t raytrace(vec_t origin, vec_t dir, int iter){
    vec_t col = {0, 0, 0};

    if(iter == BOUNCES)
        return col;
    
    int index = -1;
    float dist = FLT_MAX;
    float t = FLT_MAX;
    int norm_dir = 0;
    //check nearest triangle
    #if USE_BVH == 1
    bvh_traverse(0, &origin, &dir, &norm_dir, &t, &index);
    #else
    for(int i = 0; i < triangles_len; i++){
        int norm_tmp;
        float t_tmp = hit_triangle(&origin, &dir, &triangles[i], &norm_tmp);
        if(t_tmp > EPSILON){
            vec_t dir_scaled = vec_mul(&dir, t_tmp);
            vec_t intersection = vec_add(&origin, &dir_scaled);
            float d = vec_dist(&origin, &intersection);
            if(d < dist){
                index = i;
                dist = d;
                t = t_tmp;
                norm_dir = norm_tmp;
            }
        }
    }
    #endif
    
    if(index == -1){
        col.r += amb_light.r;
        col.g += amb_light.g;
        col.b += amb_light.b;
    } else {
        vec_t dir_scaled = vec_mul(&dir, t);
        vec_t intersection = vec_add(&origin, &dir_scaled);
        vec_t ks = triangles[index].ks;
        vec_t kd = triangles[index].kd;
        vec_t kr = triangles[index].kr;
        vec_t n = triangles[index].norm[norm_dir];
        //apply ambient light
        col.r += kd.r*amb_light.r;
        col.g += kd.g*amb_light.g;
        col.b += kd.b*amb_light.b;
        dir = vec_mul(&dir, -1.0f);
        //apply point lights
        for(int i = 0; i < lights_len; i++){
            vec_t l = vec_sub(&lights[i].pos, &intersection);
            float mag = vec_mag(&l);
            l = vec_div(&l, mag);
            mag *= mag;
            float n_dot_l = vec_dot(&n, &l);
            vec_t col_ray = lambert_blinn(&ks, &kd, &n, &l, &dir, n_dot_l);
            int V = light_v(&intersection, &l, &n, &lights[i].pos);
            col.r += V*lights[i].kl.r*col_ray.r/mag;
            col.g += V*lights[i].kl.g*col_ray.g/mag;
            col.b += V*lights[i].kl.b*col_ray.b/mag;
        }
        
        //real raytracing EXTREMELY HEAVY
        dir = vec_mul(&dir, -1);
        vec_t n_scaled = vec_mul(&n, 2*fabsf(vec_dot(&dir, &n)));
        vec_t r = vec_add(&dir, &n_scaled);
        vec_normalize(&r);
        
        if(vec_mag(&kr) > 0.0){
            vec_t col_ray = raytrace(intersection, r, iter+1);
            col.r += kr.r*col_ray.r;
            col.g += kr.g*col_ray.g;
            col.b += kr.b*col_ray.b;
        }
    }

    return col;
}