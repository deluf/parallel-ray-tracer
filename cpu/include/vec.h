#ifndef __VEC_H__
#define __VEC_H__

typedef struct vec_t {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        struct {
            float r;
            float g;
            float b;
        };
    };
} vec_t;

float vec_dot(const vec_t* v1, const vec_t* v2);
float vec_dist(const vec_t* v1, const vec_t* v2);
float vec_mag(const vec_t* v1);
void vec_normalize(vec_t* v1);
vec_t vec_mul(const vec_t* v1, float val);
vec_t vec_add(const vec_t* v1, const vec_t* v2);
vec_t vec_sub(const vec_t* v1, const vec_t* v2);
vec_t vec_div(const vec_t* v1, float val);
vec_t vec_cross(const vec_t* v1, const vec_t* v2);
void vec_constrain(vec_t* v, const vec_t* min, const vec_t* max);

#endif