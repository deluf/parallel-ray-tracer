#include <stdio.h>   // printf(), fprintf(), stderr
#include <math.h>    // sqrt()
#include <time.h>    // time_t, time()
#include <stdlib.h>  // malloc(), free(), qsort(), atoi(), exit(), RAND_MAX, EXIT_FAILURE

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
#    define M_PI 3.14159265358979323846f
#endif

cam_t cam;

size_t triangles_len;
triangle_t* triangles;

mat_t* mats;

size_t lights_len;
light_t* lights;

vec_t amb_light = vec_t
{
    0.5f,
    0.5f,
    0.5f,
    0.0f
};

vec_t pixels[WIDTH * HEIGHT];

/**
 * Comparator function for sorting float values
 * 
 * @param a Pointer to the first value
 * @param b Pointer to the second value
 * @return Positive if first is greater, negative if less, zero if equal
 */
static int compare_floats(const void* a, const void* b) 
{
    float diff = *(const float*)a - *(const float*)b;
    return (diff > 0.0f) - (diff < 0.0f);
}

/**
 * Computes the median value of a list of execution times
 * 
 * @param times Array of execution times in milliseconds
 * @param count Number of elements in the array
 * @return The median execution time
 */
static float compute_median(float times[], int count) 
{
    float* sorted_times = (float*)malloc(count * sizeof(float));
    if (!sorted_times)
    {
        fprintf(stderr, "Error: unable to allocate memory for sorted times\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < count; i++) 
    {
        sorted_times[i] = times[i];
    }
    qsort(sorted_times, count, sizeof(float), compare_floats);
    float median;
    if (count % 2 == 0) 
    {
        median = (sorted_times[count / 2 - 1] + sorted_times[count / 2]) / 2.0f;
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
static float compute_mean(float times[], int count) 
{
    float sum = 0.0f;
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
static float compute_stddev(float times[], int count, float mean) 
{
    float sum = 0.0f;
    for (int i = 0; i < count; i++) 
    {
        float diff = times[i] - mean;
        sum += diff * diff;
    }
    return sqrtf(sum / count);
}

/**
 * Computes the 99% confidence interval offset of a list of execution times
 * 
 * @param mean The mean of the execution times
 * @param stddev The standard deviation of the execution times
 * @param count Number of elements in the array
 * @return The confidence interval offset
 */
static float compute_ci(float mean, float stddev, int count) 
{
    float z = 2.5758293035489004f;
    float standard_error = stddev / sqrtf((float)count);
    return z * standard_error;
}

/**
 * Main entry point of the GPU parallel ray tracer
 * 
 * @param argc Number of command line arguments
 * @param argv Command line arguments array
 * @return 0 on success, non-zero on error
 */
int main(int argc, char** argv) 
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <tx> <ty>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int tx = atoi(argv[1]);
    int ty = atoi(argv[2]);
    if (tx <= 0 || ty <= 0)
    {
        fprintf(stderr, "Error: block dimensions must be positive integers\n");
        exit(EXIT_FAILURE);
    }

    vec_t cam_pos = vec_t
    {
        0.0f,
        -9.0f,
        3.0f,
        0.0f
    };
    vec_t cam_rot = vec_t
    {
        -M_PI / 12.0f,
        0.0f,
        0.0f,
        0.0f
    };
    if (cam_init(&cam, &cam_pos, &cam_rot, (float)(M_PI / 3.2), (float)WIDTH / HEIGHT) != 0)
    {
        fprintf(stderr, "Error: unable to initialize camera\n");
        return -1;
    }

    printf("Loading scene...\n");

    triangles = triangles_load("../assets/" SCENE "/triangles.obj", "../assets/" SCENE "/triangles.mtl", &triangles_len, &mats);
    if (!triangles)
    {
        fprintf(stderr, "Error: unable to load triangles\n");
        return -1;
    }
    
    lights = load_lights("../assets/" SCENE "/lights.obj", &lights_len);
    if (!lights && lights_len > 0)
    {
        fprintf(stderr, "Error: unable to load lights\n");
        free(triangles);
        free(mats);
        return -1;
    }

    printf("Building BVH...\n");
    bvh_build(triangles, triangles_len);

    printf("\n# Scene complexity #\n");
    printf("Resolution: %d x %d\n", WIDTH, HEIGHT);
    printf("Number of triangles: %zu\n", triangles_len);
    printf("Number of lights: %zu\n", lights_len);
    printf("Number of ray bounces: %d\n", BOUNCES);

    printf("\nRendering...\n");

    float times[ITERATIONS];

    load_to_gpu();
    for (int i = 0; i < WARMUP; i++)
    {
        render_frame(false, tx, ty);
    }
    for (int i = 0; i < ITERATIONS; i++)
    {
        times[i] = render_frame(true, tx, ty);
    }
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
    printf("Expected FPS: %.3f\n", 1000.0f / mean);

    if (bmp_write_file(pixels, WIDTH, HEIGHT, "render.bmp") != 0)
    {
        fprintf(stderr, "Error: unable to write output BMP file\n");
        free(triangles);
        free(lights);
        free(mats);
        return -1;
    }

    free(triangles);
    free(lights);
    free(mats);
    
    return 0;
}
