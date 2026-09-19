#include "lighttoken.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_all(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return NULL; }
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) { fclose(file); return NULL; }
    char *buffer = (char *)malloc((size_t)size + 1u);
    if (!buffer) { fclose(file); return NULL; }
    size_t read = fread(buffer, 1u, (size_t)size, file);
    fclose(file);
    if (read != (size_t)size) { free(buffer); return NULL; }
    buffer[size] = '\0';
    return buffer;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: lighttoken-c-smoke <token.json>\n");
        return 2;
    }
    if (lighttoken_abi_version() != LIGHTTOKEN_ABI_VERSION) return 3;

    char *token = read_all(argv[1]);
    if (!token) return 4;
    lighttoken_context *context = lighttoken_context_new();
    if (!context) { free(token); return 5; }

    char *output = NULL;
    if (lighttoken_validate_json(context, token, &output) != LIGHTTOKEN_OK || !output) return 6;
    lighttoken_string_free(output);
    output = NULL;

    if (lighttoken_compare_json(context, token, token, "cosine", &output) != LIGHTTOKEN_OK || !output) return 7;
    lighttoken_string_free(output);
    output = NULL;

    if (lighttoken_backend_json(context, &output) != LIGHTTOKEN_OK || !output) return 8;
    lighttoken_string_free(output);

    lighttoken_context_free(context);
    free(token);
    return 0;
}
