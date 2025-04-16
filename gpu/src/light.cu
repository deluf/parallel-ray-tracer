#include "light.cuh"

#include <stdio.h>
#include <stdlib.h>

light_t* lights_load(const char* filename, int* size){
    light_t* lights = NULL;
    *size = 0;
    FILE* fptr = fopen(filename, "r");
    if(!fptr){
        printf("cannot open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    lights = (light_t*)malloc(sizeof(light_t));

    char line[256];
    while (fgets(line, sizeof(line), fptr) != NULL) {
        light_t l;
        sscanf(line, "%f %f %f %f %f %f", &l.pos.x, &l.pos.y, &l.pos.z, &l.kl.r, &l.kl.g, &l.kl.b);
        *size += 1;
        lights = (light_t*)realloc(lights, sizeof(light_t)*(*size));
        lights[*size-1] = l;
    }

    fclose(fptr);

    return lights;
}