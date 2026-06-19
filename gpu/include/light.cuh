#ifndef __LIGHT_CUH__
#define __LIGHT_CUH__

#include <stddef.h>  // size_t

#include "vec.cuh"

typedef struct light_t
{
    vec_t position;
    vec_t kl;
}
light_t;

/**
 * Loads point lights from a file
 *
 * @param filename Path to the lights file
 * @param count Pointer to store the number of loaded lights
 * @return Pointer to dynamically allocated array of lights, or NULL on failure
 */
light_t* load_lights(const char* filename, size_t* count);

#endif
