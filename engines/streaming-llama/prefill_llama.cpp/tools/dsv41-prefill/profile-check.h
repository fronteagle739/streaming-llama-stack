#pragma once
#include "llama-dsv41-expert-profile.h"
#include <cstdio>
#include <stdexcept>

inline void slstack_native_profile_check() {
        namespace profile=llama_dsv41_expert_profile;
        profile::policy native;
        native.profile=profile::parse("native_half_q4k_perf_v1");
        native.admitted_pool_bytes=384ull<<20;
        native.admitted_code_runtime_bytes=2ull<<20;
        profile::validate(native,4096,nullptr);
        if(!profile::natural_half(native)||!profile::performance(native))
            throw std::runtime_error("Native producer profile policy is not compiled in");
        profile::policy rounded;
        rounded.profile=profile::parse("q8_1_activation_rounding");
        rounded.admitted_pool_bytes=320ull<<20;
        rounded.admitted_code_runtime_bytes=2ull<<20;
        profile::validate(rounded,4096,nullptr);
        if(!profile::activation_rounding_q8_1(rounded))
            throw std::runtime_error("Rounded producer profile policy is not compiled in");
        std::puts("PASS: delivered native-half and rounded expert profiles,4096-row resource policies; no GPU/model initialized");
}
