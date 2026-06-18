#ifndef LIGHT_H
#define LIGHT_H

#include <stddef.h> // size_t
#include "vec.h"

/**
 * The light_t structure represents a point light, i.e.,
 *  a point in space irradiating a light of intensity kl
 *  in all directions
 */
typedef struct 
{
    vec_t position;
    vec_t kl;   
} 
light_t;

/**
 * Load point lights from a file.
 * Each line should contain the following floats:
 *  pos_x pos_y pos_z r g b
 * 
 * @param filename Path to the lights file
 * @param count Output parameter for the number of lights loaded
 * @return Pointer to dynamically allocated array of lights, or NULL on failure
 */
light_t* load_lights(const char* filename, size_t* count);

#endif
