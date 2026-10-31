#ifndef QWEN_JSON_H
#define QWEN_JSON_H

#include <stddef.h>

/* Minimal JSON reader for model configuration files (config.json,
 * model_index.json, scheduler_config.json, safetensors index metadata).
 * Objects preserve insertion order so duplicate keys resolve to the first
 * entry, mirroring typical config semantics closely enough for these
 * single-valued files. */

typedef enum {
    QWEN_JSON_NULL = 0,
    QWEN_JSON_BOOL,
    QWEN_JSON_NUMBER,
    QWEN_JSON_STRING,
    QWEN_JSON_ARRAY,
    QWEN_JSON_OBJECT
} qw_json_kind;

typedef struct qw_json qw_json;

/* Parse NUL-terminated text. Returns 0 and fills *out on success. */
int qw_json_parse(const char *text, qw_json **out, char *error,
                  size_t error_size);
void qw_json_free(qw_json *value);

qw_json_kind qw_json_type(const qw_json *value);
const qw_json *qw_json_get(const qw_json *value, const char *key);
const qw_json *qw_json_at(const qw_json *value, size_t index);
size_t qw_json_count(const qw_json *value);
/* True bool, or number != 0, or string "true"/"1". */
int qw_json_bool(const qw_json *value);
double qw_json_number(const qw_json *value);
/* NUL-terminated UTF-8 payload of a string value (or ""). */
const char *qw_json_string(const qw_json *value);
/* Convenience: object member as double/string with fallback. */
double qw_json_get_number(const qw_json *value, const char *key,
                          double fallback);
const char *qw_json_get_string(const qw_json *value, const char *key,
                               const char *fallback);
int qw_json_get_bool(const qw_json *value, const char *key, int fallback);

#endif
