#ifndef LIGHTTOKEN_H
#define LIGHTTOKEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LIGHTTOKEN_ABI_VERSION 1u

typedef struct lighttoken_context lighttoken_context;

enum lighttoken_status {
    LIGHTTOKEN_OK = 0,
    LIGHTTOKEN_INVALID_ARGUMENT = 1,
    LIGHTTOKEN_INVALID_TOKEN = 2,
    LIGHTTOKEN_IO = 3,
    LIGHTTOKEN_UNSUPPORTED_VERSION = 4,
    LIGHTTOKEN_BACKEND = 5,
    LIGHTTOKEN_PANIC_CONTAINED = 255
};

uint32_t lighttoken_abi_version(void);
lighttoken_context *lighttoken_context_new(void);
void lighttoken_context_free(lighttoken_context *context);
int32_t lighttoken_validate_json(lighttoken_context *context, const char *json_utf8, char **out_json);
int32_t lighttoken_compare_json(lighttoken_context *context, const char *left_json_utf8,
                               const char *right_json_utf8, const char *method_utf8,
                               char **out_json);
int32_t lighttoken_search_json(lighttoken_context *context, const char *query_json_utf8,
                              const char *collection_json_utf8, const char *request_json_utf8,
                              char **out_json);
int32_t lighttoken_backend_json(lighttoken_context *context, char **out_json);
void lighttoken_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif
