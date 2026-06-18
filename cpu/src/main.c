// Enable POSIX features (e.g., pthreads, real-time clock)
#define _POSIX_C_SOURCE 199506L

#include <stdio.h>      // printf(), fprintf(), stderr, stdout
#include <math.h>       // sqrt()
#include <time.h>       // time_t, time(), clock_gettime(), struct timespec, CLOCK_MONOTONIC
#include <stdlib.h>     // rand(), srand(), malloc(), free(), qsort(), atoi(), exit(), RAND_MAX, EXIT_FAILURE
#include <pthread.h>    // pthread_t, pthread_create(), pthread_join()
#include <stdatomic.h>  // atomic_int, atomic_fetch_add()

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
void render_frame(void);
vec_t render_pixel(const vec_t* start, const vec_t* inc_x, const vec_t* inc_y, int x, int y);

int NUM_THREADS = 1;

cam_t cam;

size_t triangles_len;
triangle_t* triangles;

size_t lights_len;
light_t* lights;

vec_t amb_light = {.r = 0.5f, .g = 0.5f, .b = 0.5f};

vec_t pixels[WIDTH * HEIGHT];
atomic_int pixel_counter;

/**
 * Comparator function for sorting double values
 * 
 * @param a Pointer to the first value
 * @param b Pointer to the second value
 * @return Positive if first is greater, negative if less, zero if equal
 */
static int compare_doubles(const void* a, const void* b)
{
    double diff = *(const double*)a - *(const double*)b;
    return (diff > 0.0) - (diff < 0.0);
}

/**
 * Computes the median value of a list of execution times
 * 
 * @param times Array of execution times in milliseconds
 * @param count Number of elements in the array
 * @return The median execution time
 */
static double compute_median(double times[], int count)
{
    double* sorted_times = malloc(count * sizeof(double));
    if (!sorted_times)
    {
        fprintf(stderr, "Error: unable to allocate memory for sorted times\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < count; i++)
    {
        sorted_times[i] = times[i];
    }
    
    qsort(sorted_times, count, sizeof(double), compare_doubles);
    
    double median;
    if (count % 2 == 0)
    {
        median = (sorted_times[count / 2 - 1] + sorted_times[count / 2]) / 2.0;
    }
    else
    {
        median = sorted_times[count / 2];
    }
    
    free(sorted_times);
    return median;
}

/**
 * Computes the arithmetic mean of a list of execution times
 * 
 * @param times Array of execution times in milliseconds
 * @param count Number of elements in the array
 * @return The mean execution time
 */
static double compute_mean(double times[], int count)
{
    double sum = 0.0;
    for (int i = 0; i < count; i++)
    {
        sum += times[i];
    }
    return sum / count;
}

/**
 * Computes the standard deviation of a list of execution times
 * 
 * @param times Array of execution times in milliseconds
 * @param count Number of elements in the array
 * @param mean The mean of the execution times
 * @return The standard deviation
 */
static double compute_stddev(double times[], int count, double mean)
{
    double sum = 0.0;
    for (int i = 0; i < count; i++)
    {
        double diff = times[i] - mean;
        sum += diff * diff;
    }
    return sqrt(sum / count);
}

/**
 * Computes the 99% confidence interval offset of a list of execution times
 * 
 * @param mean The mean of the execution times
 * @param stddev The standard deviation of the execution times
 * @param count Number of elements in the array
 * @return The confidence interval offset
 */
static double compute_ci(double mean, double stddev, int count)
{
    double z = 2.5758293035489004; // 99% CI
    double standard_error = stddev / sqrt(count);
    return z * standard_error;
}

/**
 * Main entry point of the parallel ray tracer program
 * 
 * @param argc Number of command line arguments
 * @param argv Command line arguments array
 * @return 0 on success, non-zero on error
 */
int main(int argc, char* argv[])
{
#if SEED == 0
    srand((unsigned int)time(NULL));
#else
    srand(SEED);
#endif

    if (argc > 1)
    {
        NUM_THREADS = atoi(argv[1]);
        if (NUM_THREADS <= 0 || NUM_THREADS >= 64)
        {
            fprintf(stderr, "Error: invalid number of threads\n");
            exit(EXIT_FAILURE);
        }
    }

    vec_t rotation = {.x = 0.0f, .y = -M_PI / 12.0f, .z = 0.0f};
    if (cam_init(&cam, &(vec_t){0.0f, -9.0f, 2.75f}, &rotation, (float)(M_PI / 3.2), (float)WIDTH / HEIGHT) != 0)
    {
        fprintf(stderr, "Error: unable to initialize camera\n");
        return -1;
    }

    printf("Loading scene...\n");

    if (argc != 3)
    {
        triangles = triangles_load("../assets/" SCENE "/triangles.obj", "../assets/" SCENE "/triangles.mtl", &triangles_len);
        lights = load_lights("../assets/" SCENE "/lights.obj", &lights_len);
    }
    else
    {
        triangles_len = atoi(argv[2]);
        triangles = (triangle_t*)malloc(sizeof(triangle_t) * triangles_len);
        if (!triangles)
        {
            fprintf(stderr, "Error: unable to allocate memory for triangles\n");
            return -1;
        }
        for (int i = 0; i < (int)triangles_len; i++)
        {
            vec_t vec0 = {0.0f, 0.0f, 0.0f};
            vec_t vec1 = {1.0f, 1.0f, 1.0f};
            vec_t r0 = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
            vec_t r1 = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
            vec_t r2 = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
            vec_t a = vec_mul(&r0, 10.0f);
            a.x -= 5.0f; 
            a.y -= 5.0f; 
            a.z -= 5.0f;
            vec_t b = vec_add(&a, &r1);
            vec_t c = vec_add(&b, &r2);
            triangle_init(&triangles[i], &a, &b, &c, &vec1, &vec0, &vec0);
        }
        lights_len = 0;
    }

#if USE_BVH == 1
    printf("Building BVH...\n");
    struct timespec bvh_start, bvh_finish;
    clock_gettime(CLOCK_MONOTONIC, &bvh_start);

    bvh_build(triangles, triangles_len);

    clock_gettime(CLOCK_MONOTONIC, &bvh_finish);

    double bvh_elapsed = (bvh_finish.tv_sec - bvh_start.tv_sec);
    bvh_elapsed += (bvh_finish.tv_nsec - bvh_start.tv_nsec) / 1000000000.0;
    float bvh_time = (float)(bvh_elapsed * 1000.0);

    printf("bvh built in %.3f ms\n", bvh_time);
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

    for (int i = 0; i < ITERATIONS; i++)
    {
        struct timespec render_start, render_finish;
        clock_gettime(CLOCK_MONOTONIC, &render_start);

        render_frame();

        clock_gettime(CLOCK_MONOTONIC, &render_finish);

        double elapsed = (render_finish.tv_sec - render_start.tv_sec);
        elapsed += (render_finish.tv_nsec - render_start.tv_nsec) / 1000000000.0;
        times[i] = elapsed * 1000.0;

        printf("Iteration %d completed in %.3f ms\n", i + 1, times[i]);
    }

    free(triangles);
    free(lights);
    
    if (bmp_write_file(pixels, WIDTH, HEIGHT, SCENE ".bmp") != 0)
    {
        fprintf(stderr, "Error: unable to write output BMP file\n");
        return -1;
    }

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
    printf("Expected FPS: %.3f\n", 1000.0 / mean);
    
    return 0;
}

/**
 * Spawns rendering threads to render a single frame
 */
void render_frame(void)
{
    pthread_t* threads = malloc(NUM_THREADS * sizeof(pthread_t));
    if (!threads)
    {
        fprintf(stderr, "Error: unable to allocate memory for thread handles\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        if (pthread_create(&threads[i], NULL, thread_render, NULL) != 0)
        {
            fprintf(stderr, "Error: unable to create render thread %d\n", i);
            free(threads);
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    pixel_counter = 0;
}

/**
 * Renders a single pixel by casting a ray from the camera
 * 
 * @param start Top-left viewport vector coordinate
 * @param inc_x Increment vector along the X axis
 * @param inc_y Increment vector along the Y axis
 * @param x Pixel X coordinate
 * @param y Pixel Y coordinate
 * @return Color vector computed for the pixel
 */
vec_t render_pixel(const vec_t* start, const vec_t* inc_x, const vec_t* inc_y, int x, int y)
{
    vec_t dir = vec_sub(start, &cam.position);
    vec_t pos_x = vec_mul(inc_x, (float)x);
    vec_t pos_y = vec_mul(inc_y, (float)y);
    dir = vec_add(&dir, &pos_x);
    dir = vec_add(&dir, &pos_y);
    
    vec_t col = raytrace(cam.position, dir, 0);
    const vec_t vec_0 = {0.0f, 0.0f, 0.0f};
    const vec_t vec_1 = {1.0f, 1.0f, 1.0f};
    vec_constrain(&col, &vec_0, &vec_1);
    return col;
}

/**
 * Thread execution callback to render blocks of pixels dynamically
 * 
 * @param arg Unused argument
 * @return NULL
 */
void* thread_render(void* arg)
{
    (void)arg;
    vec_t ul = cam.viewport.top_left;
    vec_t ur = cam.viewport.top_right;
    vec_t dl = cam.viewport.bottom_left;
    
    vec_t inc_x = vec_sub(&ur, &ul);
    inc_x = vec_div(&inc_x, (float)WIDTH);
    vec_t inc_y = vec_sub(&dl, &ul);
    inc_y = vec_div(&inc_y, (float)HEIGHT);

    while (1)
    {
        int start_idx = atomic_fetch_add(&pixel_counter, TILE_SIZE);
        if (start_idx >= WIDTH * HEIGHT)
        {
            break;
        }
        for (int idx = start_idx; idx < start_idx + TILE_SIZE && idx < WIDTH * HEIGHT; idx++)
        {
            int x = (idx % WIDTH);
            int y = (idx / WIDTH);
            pixels[idx] = render_pixel(&ul, &inc_x, &inc_y, x, y);
        }
    }

    return NULL;
}