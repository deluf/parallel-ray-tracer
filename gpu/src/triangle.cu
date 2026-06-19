#include "triangle.cuh"
#include "vec.cuh"

#include <stdio.h>   // FILE, fopen(), sscanf(), fgets(), fclose(), fprintf(), stderr
#include <string.h>  // strcpy(), strcmp(), strncmp()
#include <stdlib.h>  // malloc(), realloc(), free(), exit(), EXIT_FAILURE

void triangle_init(triangle_t* t, const vec_t* a, const vec_t* b, const vec_t* c, int mat_idx)
{
    t->coords[0] = *a;
    t->coords[1] = *b;
    t->coords[2] = *c;
    t->mat_idx = mat_idx;

    vec_t e1 = vec_sub(&t->coords[1], &t->coords[0]);
    vec_t e2 = vec_sub(&t->coords[2], &t->coords[0]);
    t->norm[0] = vec_cross(&e1, &e2);
    vec_normalize(&t->norm[0]);
    t->norm[1] = vec_cross(&e2, &e1);
    vec_normalize(&t->norm[1]);

    t->centroid[0] = (t->coords[0].x + t->coords[1].x + t->coords[2].x) / 3.0f;
    t->centroid[1] = (t->coords[0].y + t->coords[1].y + t->coords[2].y) / 3.0f;
    t->centroid[2] = (t->coords[0].z + t->coords[1].z + t->coords[2].z) / 3.0f;
}

/**
 * Loads strings from a file
 *
 * @param filename Path to the file
 * @param lineCount Pointer to store the number of lines read
 * @return Array of pointers to the loaded strings
 */
static char** load_strings(const char* filename, int* lineCount) 
{
    FILE* file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: cannot load %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    char** lines = NULL;
    char buffer[256];
    int count = 0;
    
    while (fgets(buffer, sizeof(buffer), file)) 
    {
        lines = (char**)realloc(lines, (count + 1) * sizeof(char*));
        lines[count] = (char*)malloc(sizeof(buffer));
        strcpy(lines[count], buffer);
        count++;
    }
    
    fclose(file);
    *lineCount = count;
    return lines;
}

typedef struct 
{
    char name[256];
    vec_t kd;
    vec_t ks;
    vec_t kr;
} 
material_t;

/**
 * Parses materials from mtl lines
 *
 * @param mtl_lines Array of strings from the mtl file
 * @param mtl_line_count Number of lines in the mtl file
 * @param materials Pointer to the array to store materials
 * @param max_materials Maximum number of materials to parse
 * @return The count of parsed materials
 */
static int parse_materials(const char** mtl_lines, int mtl_line_count, material_t* materials, int max_materials) 
{
    int mat_count = 0;

    for (int i = 0; i < mtl_line_count; i++) 
    {
        if (strncmp(mtl_lines[i], "newmtl", 6) == 0 && mat_count < max_materials) 
        {
            sscanf(mtl_lines[i], "newmtl %255s", materials[mat_count].name);
            for (int j = i + 1; j < i + 6 && j < mtl_line_count; j++) 
            {
                if (strncmp(mtl_lines[j], "Kd", 2) == 0)
                {
                    sscanf(mtl_lines[j], "Kd %f %f %f", &materials[mat_count].kd.r, &materials[mat_count].kd.g, &materials[mat_count].kd.b);
                }
                else if (strncmp(mtl_lines[j], "Ks", 2) == 0)
                {
                    sscanf(mtl_lines[j], "Ks %f %f %f", &materials[mat_count].ks.r, &materials[mat_count].ks.g, &materials[mat_count].ks.b);
                }
                else if (strncmp(mtl_lines[j], "Kr", 2) == 0)
                {
                    sscanf(mtl_lines[j], "Kr %f %f %f", &materials[mat_count].kr.r, &materials[mat_count].kr.g, &materials[mat_count].kr.b);
                }
            }
            mat_count++;
        }
    }
    return mat_count;
}

triangle_t* triangles_load(const char* objname, const char* mtlname, size_t* size, mat_t** mats) 
{
    int obj_line_count, mtl_line_count;
    char** obj_lines = load_strings(objname, &obj_line_count);
    char** mtl_lines = load_strings(mtlname, &mtl_line_count);

    vec_t* vertices = (vec_t*)malloc(obj_line_count * sizeof(vec_t));
    int v_count = 0;

    for (int i = 0; i < obj_line_count; i++) 
    {
        if (obj_lines[i][0] == 'v' && obj_lines[i][1] == ' ') 
        {
            sscanf(obj_lines[i], "v %f %f %f", &vertices[v_count].x, &vertices[v_count].y, &vertices[v_count].z);
            v_count++;
        }
    }

    material_t materials[256];
    int mat_count = parse_materials((const char**)mtl_lines, mtl_line_count, materials, 256);

    mat_t* mat_array = (mat_t*)malloc(mat_count * sizeof(mat_t));
    for (int m = 0; m < mat_count; m++) 
    {
        mat_array[m].kd = materials[m].kd;
        mat_array[m].ks = materials[m].ks;
        mat_array[m].kr = materials[m].kr;
    }

    int current_mat_idx = 0;
    triangle_t* triangles = (triangle_t*)malloc(obj_line_count * sizeof(triangle_t));
    int tri_count = 0;

    for (int i = 0; i < obj_line_count; i++) 
    {
        if (strncmp(obj_lines[i], "usemtl", 6) == 0) 
        {
            char matname[256];
            sscanf(obj_lines[i], "usemtl %255s", matname);

            for (int m = 0; m < mat_count; m++) 
            {
                if (strcmp(matname, materials[m].name) == 0) 
                {
                    current_mat_idx = m;
                    break;
                }
            }
        } 
        else if (obj_lines[i][0] == 'f') 
        {
            int v1, v2, v3;
            sscanf(obj_lines[i], "f %d %d %d", &v1, &v2, &v3);
            triangle_init(&triangles[tri_count], &vertices[v1 - 1], &vertices[v2 - 1], &vertices[v3 - 1], current_mat_idx);
            tri_count++;
        }
    }

    free(vertices);
    for (int i = 0; i < obj_line_count; i++) 
    {
        free(obj_lines[i]);
    }
    free(obj_lines);

    for (int i = 0; i < mtl_line_count; i++) 
    {
        free(mtl_lines[i]);
    }
    free(mtl_lines);

    *size = tri_count;
    *mats = mat_array;

    return (triangle_t*)realloc(triangles, tri_count * sizeof(triangle_t));
}
