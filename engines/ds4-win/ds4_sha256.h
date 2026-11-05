/* Shared DS4P SHA-256 implementation, also used for direction identity. */
#ifndef DS4_SHA256_H
#define DS4_SHA256_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} peer_sha256;

static const uint32_t peer_sha256_round_constants[64] = {
    UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
    UINT32_C(0x3956c25b), UINT32_C(0x59f111f1), UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
    UINT32_C(0xd807aa98), UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
    UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
    UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786), UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
    UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
    UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
    UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147), UINT32_C(0x06ca6351), UINT32_C(0x14292967),
    UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
    UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
    UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b), UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
    UINT32_C(0xd192e819), UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
    UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
    UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a), UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
    UINT32_C(0x748f82ee), UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
    UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2),
};

static uint32_t peer_sha256_rotr(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32u - count));
}

static void peer_sha256_block(peer_sha256 *ctx, const uint8_t block[64]) {
    uint32_t words[64];
    for (uint32_t i = 0; i < 16; i++) {
        const uint8_t *p = block + (size_t)i * 4u;
        words[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                   ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    }
    for (uint32_t i = 16; i < 64; i++) {
        const uint32_t s0 = peer_sha256_rotr(words[i - 15], 7) ^
                            peer_sha256_rotr(words[i - 15], 18) ^
                            (words[i - 15] >> 3);
        const uint32_t s1 = peer_sha256_rotr(words[i - 2], 17) ^
                            peer_sha256_rotr(words[i - 2], 19) ^
                            (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = ctx->state[0];
    uint32_t b = ctx->state[1];
    uint32_t c = ctx->state[2];
    uint32_t d = ctx->state[3];
    uint32_t e = ctx->state[4];
    uint32_t f = ctx->state[5];
    uint32_t g = ctx->state[6];
    uint32_t h = ctx->state[7];
    for (uint32_t i = 0; i < 64; i++) {
        const uint32_t s1 = peer_sha256_rotr(e, 6) ^ peer_sha256_rotr(e, 11) ^
                            peer_sha256_rotr(e, 25);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + s1 + choose + peer_sha256_round_constants[i] + words[i];
        const uint32_t s0 = peer_sha256_rotr(a, 2) ^ peer_sha256_rotr(a, 13) ^
                            peer_sha256_rotr(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void peer_sha256_init(peer_sha256 *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->state[0] = UINT32_C(0x6a09e667);
    ctx->state[1] = UINT32_C(0xbb67ae85);
    ctx->state[2] = UINT32_C(0x3c6ef372);
    ctx->state[3] = UINT32_C(0xa54ff53a);
    ctx->state[4] = UINT32_C(0x510e527f);
    ctx->state[5] = UINT32_C(0x9b05688c);
    ctx->state[6] = UINT32_C(0x1f83d9ab);
    ctx->state[7] = UINT32_C(0x5be0cd19);
}

static void peer_sha256_update(peer_sha256 *ctx, const void *data, size_t len) {
    const uint8_t *p = data;
    ctx->bytes += (uint64_t)len;
    while (len != 0) {
        const size_t room = sizeof(ctx->block) - ctx->used;
        const size_t take = len < room ? len : room;
        memcpy(ctx->block + ctx->used, p, take);
        ctx->used += take;
        p += take;
        len -= take;
        if (ctx->used == sizeof(ctx->block)) {
            peer_sha256_block(ctx, ctx->block);
            ctx->used = 0;
        }
    }
}

static void peer_sha256_final(peer_sha256 *ctx,
                              uint8_t out[32]) {
    const uint64_t bits = ctx->bytes << 3;
    ctx->block[ctx->used++] = UINT8_C(0x80);
    if (ctx->used > 56u) {
        memset(ctx->block + ctx->used, 0, sizeof(ctx->block) - ctx->used);
        peer_sha256_block(ctx, ctx->block);
        ctx->used = 0;
    }
    memset(ctx->block + ctx->used, 0, 56u - ctx->used);
    for (uint32_t i = 0; i < 8; i++) {
        ctx->block[63u - i] = (uint8_t)(bits >> (8u * i));
    }
    peer_sha256_block(ctx, ctx->block);
    for (uint32_t i = 0; i < 8; i++) {
        out[(size_t)i * 4u] = (uint8_t)(ctx->state[i] >> 24);
        out[(size_t)i * 4u + 1u] = (uint8_t)(ctx->state[i] >> 16);
        out[(size_t)i * 4u + 2u] = (uint8_t)(ctx->state[i] >> 8);
        out[(size_t)i * 4u + 3u] = (uint8_t)ctx->state[i];
    }
}

static void ds4_sha256_bytes(const void *data, size_t len,
                     uint8_t out[32]) {
    peer_sha256 ctx;
    peer_sha256_init(&ctx);
    if (len != 0 && data) peer_sha256_update(&ctx, data, len);
    peer_sha256_final(&ctx, out);
}

#endif
