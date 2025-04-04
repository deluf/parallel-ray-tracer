#include "raytacer.h"
#include "triangle.h"
#include "sphere.h"
#include "light.h"
#include <float.h>
#include <math.h>
#include <stdio.h>


extern size_t spheres_len;
extern sphere_t* spheres;

extern size_t triangles_len;
extern triangle_t* triangles;

extern size_t lights_len;
extern light_t* lights;

extern vec_t amb_light;

#define MAX_ITER 8
#define EPSILON 1e-3

static vec_t lambert_blinn(const vec_t* ks, const vec_t* kd, const vec_t* n, const vec_t* l, const vec_t* v, float dot){
    vec_t h = vec_add(l, v);
    vec_normalize(&h);
    
    float coeff = fmax(0, vec_dot(n, &h));
    
    vec_t out;
    out.r = fminf((kd->r*fmaxf(0, dot)+ks->r*coeff), 1);
    out.g = fminf((kd->g*fmaxf(0, dot)+ks->g*coeff), 1);
    out.b = fminf((kd->b*fmaxf(0, dot)+ks->b*coeff), 1);

    return out;
}

static float hit_triangle(const vec_t* origin, const vec_t* dir, const triangle_t* triangle, int* norm_dir){
    *norm_dir = 0;
    vec_t e1 = vec_sub(&triangle->coords[1], &triangle->coords[0]);
    vec_t e2 = vec_sub(&triangle->coords[2], &triangle->coords[0]);
    vec_t n = vec_cross(&e1, &e2);
    float det = -vec_dot(dir, &n);
    float invdet = 1.0/det;
    vec_t ao = vec_sub(origin, &triangle->coords[0]);
    vec_t dao = vec_cross(&ao, dir);
    float u = vec_dot(&e2, &dao)*invdet;
    float v = -vec_dot(&e1, &dao)*invdet;
    float t = vec_dot(&ao, &n)*invdet;
    if(det > 0 && t > 0 && u > 0 && v > 0 && (u+v) < 1){
        return t;  
    }

    *norm_dir = 1;
    e2 = vec_sub(&triangle->coords[1], &triangle->coords[0]);
    e1 = vec_sub(&triangle->coords[2], &triangle->coords[0]);
    n = vec_cross(&e1, &e2);
    det = -vec_dot(dir, &n);
    invdet = 1.0/det;
    ao = vec_sub(origin, &triangle->coords[0]);
    dao = vec_cross(&ao, dir);
    u = vec_dot(&e2, &dao)*invdet;
    v = -vec_dot(&e1, &dao)*invdet;
    t = vec_dot(&ao, &n)*invdet;
    if(det > 0 && t > 0 && u > 0 && v > 0 && (u+v) < 1)
      return t;  
  
    return -1;
}

static float hit_sphere(const vec_t* origin, const vec_t* dir, const sphere_t* sphere){
    float a = dir->x*dir->x + dir->y*dir->y + dir->z*dir->z;
    const vec_t* pos = &sphere->pos; 
    float b = -2*( dir->x*(pos->x-origin->x) + dir->y*(pos->y-origin->y) + dir->z*(pos->z-origin->z) );
    float c = (pos->x-origin->x)*(pos->x-origin->x) + (pos->y-origin->y)*(pos->y-origin->y) + (pos->z-origin->z)*(pos->z-origin->z) - sphere->r*sphere->r;    
    float delta = b*b-4*a*c;
    if(delta < 0)
      return -1; 
    float t1 = (-b + sqrtf(delta))/(2* a);
    float t2 = (-b - sqrtf(delta))/(2* a);
    float min_t = fminf(t1, t2);
    float max_t = fmaxf(t1, t2);
    if(min_t <= 0)
      return max_t;
    else
      return min_t;
}

// light visibility - check if the ligh reach the points or is blocked by a sphere or traingle
static int light_v(const vec_t* origin, const vec_t* dir, const vec_t* light){
    vec_t tmp = vec_sub(origin, light);
    float light_dist = vec_dot(&tmp, &tmp);
    //check nearest sphere
    for(int i = 0; i < spheres_len; i++){
        float t = hit_sphere(origin, dir, &spheres[i]);
        if(t > EPSILON){
            vec_t dir_scaled = vec_mul(dir, t);
            vec_t intersection = vec_add(origin, &dir_scaled);
            vec_t o_minus_i = vec_sub(origin, &intersection);
            if(light_dist > vec_dot(&o_minus_i, &o_minus_i) )
                return 0;
        }
    }
    //check nearest triangle
    for(int i = 0; i < triangles_len; i++){
        int dummy;
        float t = hit_triangle(origin, dir, &triangles[i], &dummy);
        if(t > EPSILON){
            vec_t dir_scaled = vec_mul(dir, t);
            vec_t intersection = vec_add(origin, &dir_scaled);
            vec_t o_minus_i = vec_sub(origin, &intersection);
            if(light_dist > vec_dot(&o_minus_i, &o_minus_i) )
                return 0;
        }
    }
    return 1;
}

vec_t raytrace(vec_t origin, vec_t dir, int iter){
    vec_t col = {0};
  
    if(iter == MAX_ITER)
        return col;
    
    int index = -1;
    float dist = FLT_MAX;
    float t = -1;
    int type = 0;
    int norm_dir = 0;
    //check nearest sphere
    for(int i = 0; i < spheres_len; i++){
        float t_tmp = hit_sphere(&origin, &dir, &spheres[i]);
        if(t_tmp > EPSILON){
            vec_t dir_scaled = vec_mul(&dir, t_tmp);
            vec_t intersection = vec_add(&origin, &dir_scaled);
            float d = vec_dist(&origin, &intersection);
            if(d < dist){
                index = i;
                dist = d;
                t = t_tmp;
                type = 0;
            }
        }
    }
    //check nearest triangle
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
                type = 1;
                norm_dir = norm_tmp;
            }
        }
    }
    
    //INDEX | T | TYPE
    //TYPE = 0 : SPHERE
    //TYPE = 1 : TRIANGLE
    
    if(index < 0){
        col.r = amb_light.r;
        col.g = amb_light.g;
        col.b = amb_light.b;
    } else {
        vec_t dir_scaled = vec_mul(&dir, t);
        vec_t intersection = vec_add(&origin, &dir_scaled);
        //if intersection is sphere
        vec_t ks;
        vec_t kd;
        vec_t kr;
        vec_t n;
        if(type == 0){
            ks = spheres[index].ks;
            kd = spheres[index].kd;
            kr = spheres[index].kr;
            n = vec_sub(&intersection, &spheres[index].pos);
            vec_normalize(&n);
        }
        if(type == 1){
            ks = triangles[index].ks;
            kd = triangles[index].kd;
            kr = triangles[index].kr;
            n = triangles[index].norm[norm_dir];
        }
        //apply ambient light
        col.r = kd.r*amb_light.r;
        col.g = kd.g*amb_light.g;
        col.b = kd.b*amb_light.b;
        dir = vec_mul(&dir, -1.0f);
        //apply point lights
        for(int i = 0; i < lights_len; i++){
            vec_t l = vec_sub(&lights[i].pos, &intersection);
            float mag = vec_mag(&l);
            l = vec_div(&l, mag);
            mag *= mag;
            float n_dot_l = vec_dot(&n, &l);
            vec_t col_ray = lambert_blinn(&ks, &kd, &n, &l, &dir, n_dot_l);
            int V = light_v(&intersection, &l, &lights[i].pos);
            col.r += V*lights[i].kl.r/mag*col_ray.r;
            col.g += V*lights[i].kl.g/mag*col_ray.g;
            col.b += V*lights[i].kl.b/mag*col_ray.b;
        }
        
        //real raytracing EXTREMELY HEAVY
        dir = vec_mul(&dir, -1);
        vec_t n_scaled = vec_mul(&n, 2*fabsf(vec_dot(&dir, &n)));
        vec_t r = vec_add(&dir, &n_scaled);
        vec_normalize(&r);
        
        vec_t col_ray = raytrace(intersection, r, iter+1);
        col.r += kr.r*col_ray.r;
        col.g += kr.g*col_ray.g;
        col.b += kr.b*col_ray.b;
    }

    vec_t vec_0 = {0, 0, 0};
    vec_t vec_1 = {1, 1, 1};
    vec_constrain(&col, &vec_0, &vec_1);
        
    return col;
}