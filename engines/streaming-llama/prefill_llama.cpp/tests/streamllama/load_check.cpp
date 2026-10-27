// Check tensor mapping with no_alloc. The streaming loader still allocates and
// reads global embedding/output tensors: run under a host-memory guard even here.
// usage: load-check <model.gguf>
#include "llama.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.no_alloc     = true;
    mparams.use_mmap     = false;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "LOAD_CHECK: FAILED\n");
        return 1;
    }
    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));
    fprintf(stderr, "LOAD_CHECK: OK %s n_params=%llu n_layer=%d\n", desc,
            (unsigned long long) llama_model_n_params(model), llama_model_n_layer(model));
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
