#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include "bmp_writer.cuh"
#include "cam.cuh"
#include "triangle.cuh"
#include "light.cuh"
#include "raytracer.cuh"
#include "vec.cuh"
#include "bvh.cuh"
#include "options.cuh"

#include "gpu.cuh"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

cam_t cam;

int triangles_len;
triangle_t* triangles;

int lights_len;
light_t* lights;

vec_t amb_light = {0.5, 0.5, 0.5};

vec_t pixels[WIDTH*HEIGHT];

int main() {
    vec_t cam_pos = {0, -9, 3};
    cam_init(&cam, &cam_pos, M_PI/3.2);
    cam.rot.x = -M_PI/12;
    //cam.rot.z = M_PI/10;

    printf("Loading scene...\n");

    // Load resources once
    triangles = triangles_load("../assets/" SCENE "/triangles.obj", "../assets/" SCENE "/triangles.mtl", &triangles_len);
    lights = lights_load("../assets/" SCENE "/lights.obj", &lights_len);

    printf("Building BVH...\n");
    bvh_build(triangles, triangles_len);

    printf("\n# Scene complexity #\n");
    printf("Resolution: %d x %d\n", WIDTH, HEIGHT);
    printf("Number of triangles: %d\n", triangles_len);
    printf("Number of lights: %d\n", lights_len);
    printf("Number of ray bounces: %d\n", BOUNCES);

    printf("\nRendering...\n");

    load_to_gpu();
    render_frame();
    load_from_gpu();

    // Only save the last image
    size_t img_len;
    void* img = bmp_write(pixels, WIDTH, HEIGHT, &img_len);
    FILE* fptr = fopen("render.bmp", "wb");
    fwrite(img, 1, img_len, fptr);
    fclose(fptr);
    free(img);

    free(triangles);
    free(lights);
    
    return 0;
}