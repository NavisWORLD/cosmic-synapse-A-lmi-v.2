#include "almi.h"
#include <stdio.h>

int main(void) {
    if (almi_abi_version() != ALMI_ABI_VERSION) {
        fprintf(stderr, "ABI version mismatch\n");
        return 10;
    }
    const char *version = almi_version();
    if (version == NULL || version[0] == '\0') {
        fprintf(stderr, "version string missing\n");
        return 11;
    }
    AlmiContext *context = almi_context_new();
    if (context == NULL) {
        fprintf(stderr, "context allocation failed\n");
        return 12;
    }
    almi_context_free(context);
    printf("A-LMI C ABI %u / native %s\n", almi_abi_version(), version);
    return 0;
}
