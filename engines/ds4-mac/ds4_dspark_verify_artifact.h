/* Diagnostic-only, shared verbatim with upstream qualification control.
 * Each append is: "DS4VR001", LE u32 start/count/vocab/flags(1), count
 * LE u32 input token IDs, count*vocab LE float32 verified logits, and one
 * vocab-wide LE float32 pre-cycle frontier row (flags bit0). All candidate
 * rows are recorded before acceptance, including subsequently rejected rows.
 * The explicit environment path must be unique to the proof being recorded.
 * No GPU read, model-state change, or persistent file handle is introduced. */
#ifndef DS4_DSPARK_VERIFY_ARTIFACT_H
#define DS4_DSPARK_VERIFY_ARTIFACT_H
static bool ds41_verify_artifact_u32(FILE *f, uint32_t v) {
    unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8),
        (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    return fwrite(b, 1, 4, f) == 4;
}
static bool ds41_verify_artifact(uint32_t start, uint32_t count, const int *tokens,
        const float *rows, const float *frontier) {
    const char *path = getenv("DS4_DSPARK_VERIFY_ARTIFACT");
    if (!path || !*path) return true;
    const uint32_t one = 1;
    FILE *f = NULL;
    bool ok = sizeof(float) == 4 && *(const unsigned char *)&one == 1;
    if (ok) f = fopen(path, "ab");
    ok = ok && f && fwrite("DS4VR001", 1, 8, f) == 8 &&
        ds41_verify_artifact_u32(f, start) && ds41_verify_artifact_u32(f, count) &&
        ds41_verify_artifact_u32(f, DS4_N_VOCAB) && ds41_verify_artifact_u32(f, 1);
    for (uint32_t i = 0; ok && i < count; ++i)
        ok = ds41_verify_artifact_u32(f, (uint32_t)tokens[i]);
    const size_t n = (size_t)count * DS4_N_VOCAB;
    if (ok) ok = fwrite(rows, sizeof(float), n, f) == n &&
        fwrite(frontier, sizeof(float), DS4_N_VOCAB, f) == DS4_N_VOCAB;
    if (f && fclose(f)) ok = false;
    if (!ok) {
        fprintf(stderr, "DSpark verifier artifact write failed: %s\n", path);
    }
    return ok;
}
#endif
