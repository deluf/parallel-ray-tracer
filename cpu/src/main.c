#include <stdio.h>
#include <math.h>
#include "cam.h"
#include "sphere.h"

cam_t cam;
size_t spheres_len;
sphere_t* spheres;

int main(){
    cam_init(&cam, &(vec_t){0, 5, 0}, M_PI/4);
    spheres = sphere_load("data/spheres.obj", &spheres_len);
    
    printf("len %d\n", spheres_len);

    for(int i = 0; i < spheres_len; i++){
        sphere_t* s = &spheres[i];
        printf("%f %f %f %f\n", s->pos.x, s->pos.y, s->pos.z, s->r);
    }
}