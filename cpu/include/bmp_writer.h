#ifndef __BMP_WRITER_H__
#define __BMP_WRITER_H__

#include "vec.h"

#include <stddef.h>

void* bmp_write(vec_t* pixels, int width, int height, size_t* size);

#endif