#include "sphere.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void sphere_init(sphere_t* sp, float r, const vec_t* pos, const vec_t* kd, const vec_t* ks){
    sp->r = r;
    sp->pos = *pos;
    sp->kd = *kd;
    sp->ks = *ks;
}

sphere_t* sphere_load(const char* filename, size_t* size){
    sphere_t* spheres = NULL;
    *size = 0;
    FILE* fptr = fopen(filename, "r");
    if(!fptr){
        printf("cannot open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    spheres = malloc(sizeof(sphere_t));

    char line[256];
    while (fgets(line, sizeof(line), fptr) != NULL) {
        sphere_t s;
        sscanf(line, "%f %f %f %f %f %f %f %f %f %f", &s.pos.x, &s.pos.y, &s.pos.z, &s.r, &s.kd.r, &s.kd.g, &s.kd.b, &s.ks.r, &s.ks.g, &s.ks.b);
        s.kr = (vec_t){0.5f, 0.5f, 0.5f};
        *size += 1;
        spheres = realloc(spheres, sizeof(sphere_t)*(*size));
        spheres[*size-1] = s;
    }

    fclose(fptr);

    return spheres;
}