
// Enable POSIX features (e.g., pthreads, real-time clock)
#define _POSIX_C_SOURCE 199506L

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>

#include "bmp_writer.h"
#include "cam.h"
#include "triangle.h"
#include "light.h"
#include "raytracer.h"
#include "vec.h"
#include "bvh.h"
#include "options.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

void* thread_render(void* arg);

int NUM_THREADS = 1;

cam_t cam;

size_t triangles_len;
triangle_t* triangles;

size_t lights_len;
light_t* lights;

vec_t amb_light = {.r = 0.5, .g = 0.5, .b = 0.5};

vec_t pixels[WIDTH*HEIGHT];
atomic_int pixel_counter;

void render_frame();
vec_t render_pixel(const vec_t* start, const vec_t* inc_x, const vec_t* inc_y, int x, int y);

int compare_doubles(const void* a, const void* b) {
    double diff = *(double*)a - *(double*)b;
    return (diff > 0) - (diff < 0);
}

double compute_median(double times[], int count) {
    double* sorted_times = malloc(count * sizeof(double));
    for (int i = 0; i < count; i++) {
        sorted_times[i] = times[i];
    }
    qsort(sorted_times, count, sizeof(double), compare_doubles);
    double median;
    if (count % 2 == 0) {
        median = (sorted_times[count / 2 - 1] + sorted_times[count / 2]) / 2.0;
    } else {
        median = sorted_times[count / 2];
    }
    free(sorted_times);
    return median;
}

double compute_mean(double times[], int count) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += times[i];
    }
    return sum / count;
}

double compute_stddev(double times[], int count, double mean) {
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = times[i] - mean;
        sum += diff * diff;
    }
    return sqrt(sum / count);
}

double compute_ci(double mean, double stddev, int count) {
    //double z = 1.959963984540054;		// 95% CI
    double z = 2.5758293035489004;		// 99% CI
    double standard_error = stddev / sqrt(count);
    return z * standard_error;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        NUM_THREADS = atoi(argv[1]);
        if (NUM_THREADS <= 0 || NUM_THREADS >= 64) {
            fprintf(stdout, "Invalid number of threads\n");
            exit(-1);
        }
    }

    cam_init(&cam, &(vec_t){0, -9, 3}, M_PI/3.2);
    cam.rot.x = -M_PI/12;
    //cam.rot.z = M_PI/10;

    printf("Loading scene...\n");

    // Load resources once
    triangles = triangles_load("../assets/" SCENE "/triangles.obj", "../assets/" SCENE "/triangles.mtl", &triangles_len);
    lights = lights_load("../assets/" SCENE "/lights.obj", &lights_len);

    #if USE_BVH == 1
    printf("Building BVH...\n");
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    bvh_build(triangles, triangles_len);

    clock_gettime(CLOCK_MONOTONIC, &finish);

    double elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    float time = elapsed * 1000; // Convert to milliseconds

    printf("bvh built in %.3f ms\n", time);

    printf("\n# BVH settings #\n");
    printf("Max depth: %d\n", BVH_MAX_ITER);
    printf("Leaf size threshold: %d\n", BVH_ELEMENT_THRESHOLD);
    printf("Split heuristic: %d\n", BVH_HEURISTIC);
    printf("Seed: %d\n", SEED);
    printf("Fast light: %d\n", USE_BVH_FAST_LIGHT);
    #endif

    printf("\n# Host settings #\n");
    printf("Number of threads: %d\n", NUM_THREADS);

    printf("\n# Scene complexity #\n");
    printf("Resolution: %d x %d\n", WIDTH, HEIGHT);
    printf("Number of triangles: %zu\n", triangles_len);
    printf("Number of lights: %zu\n", lights_len);
    printf("Number of ray bounces: %d\n", BOUNCES);


    printf("\nRendering...\n");

    double times[ITERATIONS];

    for (int i = 0; i < ITERATIONS; i++) {
        struct timespec start, finish;
        clock_gettime(CLOCK_MONOTONIC, &start);

        render_frame();

        clock_gettime(CLOCK_MONOTONIC, &finish);

        double elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        times[i] = elapsed * 1000; // Convert to milliseconds

        printf("Iteration %d completed in %.3f ms\n", i + 1, times[i]);
    }

    // Only save the last image
    size_t img_len;
    void* img = bmp_write(pixels, WIDTH, HEIGHT, &img_len);
    FILE* fptr = fopen("render.bmp", "wb");
    fwrite(img, 1, img_len, fptr);
    fclose(fptr);
    free(img);

    free(triangles);
    free(lights);

    // Compute metrics
    double mean = compute_mean(times, ITERATIONS);
    double median = compute_median(times, ITERATIONS);
    double stddev = compute_stddev(times, ITERATIONS, mean);
    double ci_offset = compute_ci(mean, stddev, ITERATIONS);
    
    printf("\n# Metrics #\n");
    printf("Total execution time of %d frames: %.3f ms\n", ITERATIONS, mean * ITERATIONS);
    #if ITERATIONS >= 30
        printf("Frame time (mean +/- 99%% CI): %.3f +/- %.3f = [%.3f, %.3f] ms\n",
            mean, ci_offset, mean - ci_offset, mean + ci_offset);
    #else
        printf("Frame time (mean): %.3f ms\n", mean);
    #endif
    printf("Frame time (median): %.3f ms\n", median);
    printf("Frame time (stddev): %.3f ms^2\n", stddev);
    printf("Expected FPS: %.3f\n", 1000 / mean);
    
    return 0;
}

void render_frame(){
    pthread_t threads[NUM_THREADS];

    for(int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, thread_render, NULL);

    for(int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    //reset pixel counter
    pixel_counter = 0;
}

vec_t render_pixel(const vec_t* start, const vec_t* inc_x, const vec_t* inc_y, int x, int y){
    vec_t dir = vec_sub(start, &cam.pos);
    vec_t pos_x = vec_mul(inc_x, x);
    vec_t pos_y = vec_mul(inc_y, y);
    dir = vec_add(&dir, &pos_x);
    dir = vec_add(&dir, &pos_y);
    vec_t col = raytrace(cam.pos, dir, 0);
    const vec_t vec_0 = {0, 0, 0};
    const vec_t vec_1 = {1, 1, 1};
    vec_constrain(&col, &vec_0, &vec_1);
    return col;
}

void* thread_render(void* arg) {
    vec_t screen_points[3];
    cam_calculate_screen_coords(&cam, screen_points, (float)WIDTH/HEIGHT);
    vec_t ul = screen_points[0];
    vec_t ur = screen_points[1];
    vec_t dl = screen_points[2];
    vec_t inc_x = vec_sub(&ur, &ul);
    inc_x = vec_div(&inc_x, WIDTH);
    vec_t inc_y = vec_sub(&dl, &ul);
    inc_y = vec_div(&inc_y, HEIGHT);

    while(1){
        int start_idx = atomic_fetch_add(&pixel_counter, TILE_SIZE);
        if(start_idx >= WIDTH*HEIGHT)
            break;
        for(int idx = start_idx; idx < start_idx + TILE_SIZE && idx < WIDTH*HEIGHT; idx++){
            int x = (idx % WIDTH);
            int y = (idx / WIDTH);
            pixels[idx] = render_pixel(&ul, &inc_x, &inc_y, x, y);
        }
    }

    return NULL;
}