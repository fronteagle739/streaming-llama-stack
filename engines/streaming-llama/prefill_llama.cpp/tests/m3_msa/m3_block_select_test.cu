// Standalone TDD harness for the MiniMax-M3 MSA block-SELECTION logic (the core of the
// ggml-cuda block-max-pool + block-topk kernels). Reads a fixture exported by
// scripts/m3_msa_reference.py --export-fixture, runs the selection on the GPU, and asserts the
// selected block ids match the numpy oracle. No model weights needed.
//
// The selection LOGIC is block_size/d-independent; this harness uses the fixture's tiny dims
// (the real kernel runs block_size=128, d=128). One thread per (head, query token); double
// precision to match the oracle's float64. Correct-by-construction (not optimized) — it is the
// reference the optimized ggml-cuda kernel must match.
//
// Build + run (from this dir, with the CUDA toolkit + MSVC env set up):
//   nvcc -O2 -std=c++17 m3_block_select_test.cu -o m3_block_select_test.exe
//   ./m3_block_select_test.exe block_select_fixture.txt
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(2); } } while (0)

#define INIT_SCORE  1e30   // forced init (sink) blocks
#define LOCAL_SCORE 1e29   // forced local (recent) blocks; init wins on overlap

// One thread per (h, t). idx layout matches SEL: [h][t].
__global__ void m3_block_select(const double* iq, const double* ik, int* sel,
                                int n, int H, int d, int bs, int topk,
                                int init_b, int local_b) {
    const int MAXB = 64;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * H) return;
    int h = idx / n;
    int t = idx % n;
    int valid = t / bs + 1;               // blocks holding any key j <= t

    double score[MAXB];
    for (int b = 0; b < valid; ++b) {
        int lo = b * bs;
        int hi = (b + 1) * bs; if (hi > t + 1) hi = t + 1;  // causal j <= t
        double mx = -1e30;
        for (int j = lo; j < hi; ++j) {
            double dot = 0.0;
            for (int c = 0; c < d; ++c) dot += iq[(t * H + h) * d + c] * ik[j * d + c];
            if (dot > mx) mx = dot;
        }
        if (b >= valid - local_b) mx = LOCAL_SCORE;   // local first
        if (b < init_b)           mx = INIT_SCORE;    // init wins on overlap
        score[b] = mx;
    }

    // top-k by (score desc, id asc); strict '>' + ascending scan keeps the lowest id on ties.
    int chosen[MAXB];
    bool used[MAXB];
    for (int b = 0; b < valid; ++b) used[b] = false;
    int k = topk < valid ? topk : valid;
    for (int s = 0; s < k; ++s) {
        int best = -1; double bestsc = -1e300;
        for (int b = 0; b < valid; ++b) {
            if (used[b]) continue;
            if (score[b] > bestsc) { bestsc = score[b]; best = b; }
        }
        used[best] = true; chosen[s] = best;
    }
    // sort the chosen k ascending (small k -> insertion)
    for (int a = 0; a < k; ++a)
        for (int b2 = a + 1; b2 < k; ++b2)
            if (chosen[b2] < chosen[a]) { int tmp = chosen[a]; chosen[a] = chosen[b2]; chosen[b2] = tmp; }
    for (int s = 0; s < topk; ++s) sel[(h * n + t) * topk + s] = (s < k) ? chosen[s] : -1;
}

static std::vector<std::string> tok(const std::string& line) {
    std::vector<std::string> v; std::istringstream is(line); std::string s;
    while (is >> s) v.push_back(s);
    return v;
}

int main(int argc, char** argv) {
    const char* path = argc >= 2 ? argv[1] : "block_select_fixture.txt";
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open fixture %s\n", path); return 2; }

    int n=0,H=0,d=0,bs=0,topk=0,init_b=0,local_b=0;
    std::vector<double> iq, ik; std::vector<int> sel_ref;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto v = tok(line);
        if (v[0] == "DIMS") {
            n=atoi(v[1].c_str()); H=atoi(v[2].c_str()); d=atoi(v[3].c_str());
            bs=atoi(v[4].c_str()); topk=atoi(v[5].c_str()); init_b=atoi(v[6].c_str()); local_b=atoi(v[7].c_str());
        } else if (v[0] == "IQ") { for (size_t i=1;i<v.size();++i) iq.push_back(atof(v[i].c_str()));
        } else if (v[0] == "IK") { for (size_t i=1;i<v.size();++i) ik.push_back(atof(v[i].c_str()));
        } else if (v[0] == "SEL") { for (size_t i=1;i<v.size();++i) sel_ref.push_back(atoi(v[i].c_str())); }
    }
    if ((int)iq.size()!=n*H*d || (int)ik.size()!=n*d || (int)sel_ref.size()!=H*n*topk) {
        fprintf(stderr, "fixture size mismatch: iq=%zu(want %d) ik=%zu(want %d) sel=%zu(want %d)\n",
                iq.size(), n*H*d, ik.size(), n*d, sel_ref.size(), H*n*topk); return 2;
    }
    printf("fixture: n=%d H=%d d=%d block_size=%d topk=%d init=%d local=%d\n", n,H,d,bs,topk,init_b,local_b);

    double *d_iq=nullptr,*d_ik=nullptr; int *d_sel=nullptr;
    CUDA_OK(cudaMalloc(&d_iq, iq.size()*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_ik, ik.size()*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_sel, sel_ref.size()*sizeof(int)));
    CUDA_OK(cudaMemcpy(d_iq, iq.data(), iq.size()*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ik, ik.data(), ik.size()*sizeof(double), cudaMemcpyHostToDevice));

    int total = n*H, threads = 64, blocks = (total + threads - 1) / threads;
    m3_block_select<<<blocks, threads>>>(d_iq, d_ik, d_sel, n, H, d, bs, topk, init_b, local_b);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<int> sel_gpu(sel_ref.size());
    CUDA_OK(cudaMemcpy(sel_gpu.data(), d_sel, sel_ref.size()*sizeof(int), cudaMemcpyDeviceToHost));

    int mism = 0;
    for (size_t i = 0; i < sel_ref.size(); ++i)
        if (sel_gpu[i] != sel_ref[i]) {
            if (mism < 10) printf("  MISMATCH at %zu: gpu=%d ref=%d\n", i, sel_gpu[i], sel_ref[i]);
            ++mism;
        }
    cudaFree(d_iq); cudaFree(d_ik); cudaFree(d_sel);
    if (mism == 0) { printf("PASS: %d (head,token) selections match the oracle\n", total); return 0; }
    printf("FAIL: %d / %zu selection entries mismatch\n", mism, sel_ref.size());
    return 1;
}
