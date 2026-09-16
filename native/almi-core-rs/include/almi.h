#ifndef ALMI_NATIVE_H
#define ALMI_NATIVE_H

#include <stdint.h>

#ifdef _WIN32
  #ifdef ALMI_NATIVE_BUILD
    #define ALMI_API __declspec(dllexport)
  #else
    #define ALMI_API __declspec(dllimport)
  #endif
#else
  #define ALMI_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ALMI_ABI_VERSION 1u

#define ALMI_OK 0
#define ALMI_ERROR_NULL 1
#define ALMI_ERROR_INVALID 3
#define ALMI_ERROR_INTEGRITY 4
#define ALMI_ERROR_SECURITY 5
#define ALMI_ERROR_UNSUPPORTED_VERSION 6
#define ALMI_ERROR_IO 7
#define ALMI_ERROR_PANIC 8
#define ALMI_ERROR_INTERNAL 9

typedef struct AlmiContext AlmiContext;

ALMI_API uint32_t almi_abi_version(void);
ALMI_API const char *almi_version(void);

/* Context owns only error state. It does not own user workspaces or model authority. */
ALMI_API AlmiContext *almi_context_new(void);
ALMI_API void almi_context_free(AlmiContext *context);
ALMI_API const char *almi_last_error(const AlmiContext *context);

/* Returns an ALMI_* error code. Paths must be non-null UTF-8 strings. */
ALMI_API int32_t almi_workspace_validate(AlmiContext *context, const char *path);

/*
 * On ALMI_OK, *out_json receives an allocated UTF-8 JSON string.
 * The caller must release it with almi_string_free().
 */
ALMI_API int32_t almi_cosmos_verify_json(AlmiContext *context, const char *path, char **out_json);
ALMI_API int32_t almi_cosmos_inspect_json(AlmiContext *context, const char *path, char **out_json);
ALMI_API void almi_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif /* ALMI_NATIVE_H */
