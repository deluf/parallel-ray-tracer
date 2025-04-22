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

mat_t* mats;

int lights_len;
light_t* lights;

vec_t amb_light = {0.5, 0.5, 0.5};

vec_t pixels[WIDTH*HEIGHT];

int compare_floats(const void* a, const void* b) {
    float diff = *(float*)a - *(float*)b;
    return (diff > 0) - (diff < 0);
}

float compute_median(float times[], int count) {
    float* sorted_times = (float*)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) {
        sorted_times[i] = times[i];
    }
    qsort(sorted_times, count, sizeof(float), compare_floats);
    float median;
    if (count % 2 == 0) {
        median = (sorted_times[count / 2 - 1] + sorted_times[count / 2]) / 2.0;
    } else {
        median = sorted_times[count / 2];
    }
    free(sorted_times);
    return median;
}

float compute_mean(float times[], int count) {
    float sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += times[i];
    }
    return sum / count;
}

float compute_stddev(float times[], int count, float mean) {
    float sum = 0.0;
    for (int i = 0; i < count; i++) {
        float diff = times[i] - mean;
        sum += diff * diff;
    }
    return sqrt(sum / count);
}

float compute_ci(float mean, float stddev, int count) {
    //float z = 1.959963984540054;		// 95% CI
    float z = 2.5758293035489004;		// 99% CI
    float standard_error = stddev / sqrt(count);
    return z * standard_error;
}

int main(int argc, char** argv) {
    if(argc != 3){
        printf("<exe> <tx> <ty>\n");
        exit(EXIT_FAILURE);
    }

    vec_t cam_pos = {0, -9, 3};
    cam_init(&cam, &cam_pos, M_PI/3.2);
    cam.rot.x = -M_PI/12;
    //cam.rot.z = M_PI/10;

    printf("Loading scene...\n");

    // Load resources once
    triangles = triangles_load("../assets/" SCENE "/triangles.obj", "../assets/" SCENE "/triangles.mtl", &triangles_len, &mats);
    lights = lights_load("../assets/" SCENE "/lights.obj", &lights_len);

    printf("Building BVH...\n");
    bvh_build(triangles, triangles_len);

    printf("\n# Scene complexity #\n");
    printf("Resolution: %d x %d\n", WIDTH, HEIGHT);
    printf("Number of triangles: %d\n", triangles_len);
    printf("Number of lights: %d\n", lights_len);
    printf("Number of ray bounces: %d\n", BOUNCES);

    printf("\nRendering...\n");

    float times[ITERATIONS];

    load_to_gpu();
    for(int i = 0; i < WARMUP; i++)
        render_frame(false, atoi(argv[1]), atoi(argv[2]));
    for(int i = 0; i < ITERATIONS; i++)
        times[i] = render_frame(true, atoi(argv[1]), atoi(argv[2]));
    load_from_gpu();

    float mean = compute_mean(times, ITERATIONS);
    float median = compute_median(times, ITERATIONS);
    float stddev = compute_stddev(times, ITERATIONS, mean);
    float ci_offset = compute_ci(mean, stddev, ITERATIONS);
    
    printf("\n# Metrics #\n");
    printf("Total execution time of %d frames: %.3f ms\n", ITERATIONS, mean * ITERATIONS);
    printf("Frame time (mean +/- 99%% CI): %.3f +/- %.3f = [%.3f, %.3f] ms\n", mean, ci_offset, mean - ci_offset, mean + ci_offset);
    printf("Frame time (median): %.3f ms\n", median);
    printf("Frame time (stddev): %.3f ms^2\n", stddev);
    printf("Expected FPS: %.3f\n", 1000 / mean);

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