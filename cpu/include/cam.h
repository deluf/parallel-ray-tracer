#ifndef __CAM_H__
#define __CAM_H__

#include "vec.h"

typedef struct cam_t {
    vec_t pos;
    vec_t rot;
    float fov;
} cam_t;

void cam_init(cam_t* cam, const vec_t* pos, float fov);
void cam_rotate(cam_t* cam, vec_t* p);
void cam_rotateX(cam_t* cam, vec_t* p);
void cam_rotateY(cam_t* cam, vec_t* p);
void cam_rotateZ(cam_t* cam, vec_t* p);
void cam_calculate_screen_coords(cam_t* cam, vec_t* vecs);

#endif