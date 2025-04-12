
// Enable POSIX features (e.g., pthreads, real-time clock)
#define _POSIX_C_SOURCE 199506L

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>

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

typedef struct {
    int from_idx;
    int to_idx;
} render_task_t;

void* thread_render(void* arg);

cam_t cam;

size_t triangles_len;
triangle_t* triangles;

size_t lights_len;
light_t* lights;

vec_t amb_light = {.r = 0.5, .g = 0.5, .b = 0.5};

vec_t pixels[WIDTH*HEIGHT];

void render_frame();
void render_segment(int from_idx, int to_idx);
vec_t render_pixel(const vec_t* start, const vec_t* inc_x, const vec_t* inc_y, int x, int y);

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
    double z = 1.959963984540054;
    double standard_error = stddev / sqrt(count);
    return z * standard_error;
}

int main() {
    cam_init(&cam, &(vec_t){0, -9, 3}, M_PI/3.2);
    cam.rot.x = -M_PI/12;

    printf("Loading scene...\n");

    // Load resources once
    triangles = triangles_load(SCENE "/triangles.obj", SCENE "/triangles.mtl", &triangles_len);
    lights = lights_load(SCENE "/lights.obj", &lights_len);

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
    #endif

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
    double stddev = compute_stddev(times, ITERATIONS, mean);
    double ci_offset = compute_ci(mean, stddev, ITERATIONS);
    
    printf("\n# Metrics #\n");
    printf("Total execution time of %d frames: %.3f ms\n", ITERATIONS, mean * ITERATIONS);
    if (ITERATIONS >= 30)
        printf("Frame time (mean +/- 95%% CI): %.3f +/- %.3f = [%.3f, %.3f] ms\n",
            mean, ci_offset, mean - ci_offset, mean + ci_offset);
    else
        printf("Frame time (mean): %.3f ms\n", mean);
    printf("Frame time (stddev): %.3f ms^2\n", stddev);
    printf("Expected FPS: %.3f\n", 1000 / mean);
    //printf("Pixel time (mean): %.3f ms\n", (mean * 1000) / (WIDTH * HEIGHT));
    
    return 0;
}

void render_frame(){
    pthread_t threads[NUM_THREADS];
    render_task_t tasks[NUM_THREADS];

    int pixels_per_thread = (WIDTH * HEIGHT) / NUM_THREADS;

    for(int i = 0; i < NUM_THREADS; i++) {
        tasks[i].from_idx = i * pixels_per_thread;
        tasks[i].to_idx = (i == NUM_THREADS - 1) ? (WIDTH * HEIGHT) : (i + 1) * pixels_per_thread;

        pthread_create(&threads[i], NULL, thread_render, &tasks[i]);
    }

    for(int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

void render_segment(int from_idx, int to_idx){
    int from_x = from_idx % WIDTH;
    int from_y = from_idx / WIDTH;
    int to_x = to_idx % WIDTH;
    int to_y = to_idx / WIDTH;
    int pixels_len = to_idx - from_idx;
    vec_t screen_points[3];
    cam_calculate_screen_coords(&cam, screen_points, (float)WIDTH/HEIGHT);
    vec_t ul = screen_points[0];
    vec_t ur = screen_points[1];
    vec_t dl = screen_points[2];
    vec_t inc_x = vec_sub(&ur, &ul);
    inc_x = vec_div(&inc_x, WIDTH);
    vec_t inc_y = vec_sub(&dl, &ul);
    inc_y = vec_div(&inc_y, HEIGHT);

    for(int i = 0; i < pixels_len; i++){
        int idx = from_idx + i;
        int x = (idx % WIDTH);
        int y = (idx / WIDTH);
        pixels[from_idx + i] = render_pixel(&ul, &inc_x, &inc_y, x, y);
    }
}

vec_t render_pixel(const vec_t* start, const vec_t* inc_x, const vec_t* inc_y, int x, int y){
    vec_t dir = vec_sub(start, &cam.pos);
    vec_t pos_x = vec_mul(inc_x, x);
    vec_t pos_y = vec_mul(inc_y, y);
    dir = vec_add(&dir, &pos_x);
    dir = vec_add(&dir, &pos_y);
    return raytrace(cam.pos, dir, 0);
}

void* thread_render(void* arg) {
    render_task_t* task = (render_task_t*)arg;
    render_segment(task->from_idx, task->to_idx);
    return NULL;
}