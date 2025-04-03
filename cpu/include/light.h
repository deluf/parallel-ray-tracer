#ifndef __LIGHT_H__
#define __LIGHT_H__

#include <stddef.h>

#include "vec.h"

typedef struct light_t {
    vec_t pos;
    vec_t kl;
} light_t;

light_t* lights_load(const char* filename, size_t* size);

#endif