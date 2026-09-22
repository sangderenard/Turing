#include <stdint.h>

static _Thread_local uint32_t turing_validation_code = 0;

void turing_validation_error(uint32_t error_code) {
    if (turing_validation_code == 0) {
        turing_validation_code = error_code == 0 ? UINT32_MAX : error_code;
    }
}

void turing_validation_error_reset(void) {
    turing_validation_code = 0;
}

uint32_t turing_validation_error_take(void) {
    uint32_t result = turing_validation_code;
    turing_validation_code = 0;
    return result;
}
