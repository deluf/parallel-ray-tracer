#include <stdio.h>
#include <math.h>

#include "bmp_writer.h"
#include "cam.h"
#include "sphere.h"
#include "triangle.h"
#include "light.h"
#include "raytacer.h"

#define WIDTH 1024
#define HEIGHT 1024

cam_t cam;

size_t spheres_len;
sphere_t* spheres;

size_t triangles_len;
triangle_t* triangles;

size_t lights_len;
light_t* lights;

vec_t amb_light = {.r = 0.1, .g = 0.1, .b = 0.1};

vec_t pixels[WIDTH*HEIGHT];

int main(){
    cam_init(&cam, &(vec_t){-9, 0, 8}, M_PI/4);
    //cam.rot.x = -M_PI/6;
    cam.rot.z = -M_PI/3;
    cam.rot.x = -M_PI/6;
    spheres = sphere_load("data/spheres.obj", &spheres_len);
    triangles = triangles_load("data/triangles.obj", "data/triangles.mtl", &triangles_len);
    lights = lights_load("data/lights.obj", &lights_len);
    
    vec_t screen_points[3];
    cam_calculate_screen_coords(&cam, screen_points);
    vec_t ul = screen_points[0];
    vec_t ur = screen_points[1];
    vec_t dl = screen_points[2];
    vec_t inc_x = vec_sub(&ur, &ul);
    inc_x = vec_div(&inc_x, WIDTH);
    vec_t inc_y = vec_sub(&dl, &ul);
    inc_y = vec_div(&inc_y, HEIGHT);
    for(int y = 0; y < HEIGHT; y++){
        for(int x = 0; x < WIDTH; x++){
            vec_t dir = vec_sub(&ul, &cam.pos);
            vec_t pos_x = vec_mul(&inc_x, x);
            vec_t pos_y = vec_mul(&inc_y, y);
            dir = vec_add(&dir, &pos_x);
            dir = vec_add(&dir, &pos_y);
            pixels[x+y*WIDTH] = raytrace(cam.pos, dir, 0);
        }
    }

    size_t img_len;
    void* img = bmp_write(pixels, WIDTH, HEIGHT, &img_len);

    FILE* fptr = fopen("img.bmp", "wb");
    fwrite(img, 1, img_len, fptr);
    fclose(fptr);

    printf("END!\n");
}