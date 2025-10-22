#ifndef __BMP_WRITER_H__
#define __BMP_WRITER_H__

#include "vec.h"

int bmp_write_file(vec_t* pixels, int width, int height, const char* filename);

#endif
