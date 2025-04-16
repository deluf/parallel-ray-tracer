#include "bmp_writer.cuh"

#include <stdlib.h>
#include <string.h>

#define RGB(r, g, b) ((b) | ((g) << 8) | ((r) << 16) | (255 << 24))

void* bmp_write(vec_t* pixels, int width, int height, size_t* size) {
    const int BITMAP_FILE_HEADER = 14;
    const int BITMAPINFOHEADER_SIZE = 40;
    const int BPP = 32;
    const int ROW_SIZE = (BPP * width + 31) / 32 * 4;
    const int PIXEL_ARRAY_SIZE = ROW_SIZE * height;
    const int FILE_SIZE = BITMAP_FILE_HEADER + BITMAPINFOHEADER_SIZE + PIXEL_ARRAY_SIZE;
    const int START_ARRAY_OFFSET = BITMAP_FILE_HEADER + BITMAPINFOHEADER_SIZE;

    unsigned char* bytes = (unsigned char*)malloc(FILE_SIZE);
    memset(bytes, 0, FILE_SIZE);

    // BMP Header
    bytes[0x00] = 'B';
    bytes[0x01] = 'M';
    memcpy(bytes + 0x02, &FILE_SIZE, 4);
    memset(bytes + 0x06, 0, 4);  // Reserved
    memcpy(bytes + 0x0A, &START_ARRAY_OFFSET, 4);

    // DIB Header (BITMAPINFOHEADER)
    memcpy(bytes + 0x0E, &BITMAPINFOHEADER_SIZE, 4);
    memcpy(bytes + 0x12, &width, 4);
    memcpy(bytes + 0x16, &height, 4);
    bytes[0x1A] = 1;  // Planes
    bytes[0x1C] = BPP;
    memset(bytes + 0x1E, 0, 4);  // Compression (0 = BI_RGB)

    unsigned char* data_ptr = bytes + START_ARRAY_OFFSET;
    int* rgb_pixels = (int*)malloc(width * height * sizeof(int));

    for(int i = 0; i < width * height; i++)
        rgb_pixels[i] = RGB((int)(pixels[i].r*255), (int)(pixels[i].g*255), (int)(pixels[i].b*255));

    for(int y = 0; y < height; y++)
        memcpy(data_ptr + y * ROW_SIZE, rgb_pixels + (height - 1 - y) * width, width * 4);

    free(rgb_pixels);
    *size = FILE_SIZE;
    return bytes;
}
