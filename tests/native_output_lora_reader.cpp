#include "ds4_lora.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    ds4_output_lora adapter = {};
    char error[256] = {};
    const int ok = ds4_output_lora_read(argv[1], argv[2],
        (uint32_t)strtoul(argv[3], nullptr, 10),
        (uint32_t)strtoul(argv[4], nullptr, 10), &adapter, error, sizeof(error));
    if (ok) printf("rank=%u alpha=%g\n", adapter.rank, adapter.alpha);
    else fprintf(stderr, "%s\n", error);
    ds4_output_lora_free(&adapter);
    return ok ? 0 : 1;
}
